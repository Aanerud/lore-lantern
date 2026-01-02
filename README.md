# Lore Lantern

Multi-agent story generation system using CrewAI, LiteLLM, and Azure services.

**https://lorelantern.com** | **https://lorelantern.no**

---

## Test-Driven Development

After deployment, verify the system is working with the E2E tests in `tests/`:

```bash
# Start server
python -m src.main &

# Run Norwegian language test (3-step conversation flow)
python tests/e2e_inger_helene_norwegian.py

# Run English Viking story test (prefetch, non-blocking dialogue)
python tests/e2e_harald_emma.py
```

**What the tests validate:**
- Character completeness (5/5 created)
- Chapter completeness (5/5 written)
- Language detection (Norwegian/English)
- Dialogue grounding (story-relevant responses)
- Non-blocking dialogue during generation

---

## Architecture

### Two-Track Design: User-Facing vs Background

```
┌─────────────────────────────────────┐     ┌─────────────────────────────────────┐
│         USER-FACING TRACK           │     │         BACKGROUND TRACK            │
│         (Never blocks)              │     │         (Heavy processing)          │
├─────────────────────────────────────┤     ├─────────────────────────────────────┤
│                                     │     │                                     │
│  USER ←──→ CompanionAgent           │     │  COORDINATOR                        │
│              │                      │     │      │                              │
│              ├─→ Immediate response │     │      ├─→ StructureAgent (CrewAI)    │
│              │   (async, main loop) │     │      ├─→ CharacterAgent (CrewAI)    │
│              │                      │     │      ├─→ NarrativeAgent (CrewAI)    │
│              ├─→ Reads Azure SQL    │     │      ├─→ Round Table (6 reviewers)  │
│              │   (sees all state)   │     │      ├─→ VoiceDirectorAgent         │
│              │                      │     │      └─→ TTS Generation             │
│              └─→ Emits events       │     │                                     │
│                  (prefetch, style)  │     │  Runs in CREWAI_EXECUTOR            │
│                                     │     │  (dedicated 8-thread pool)          │
└─────────────────────────────────────┘     └─────────────────────────────────────┘
```

### Why This Design?

**CompanionAgent is NOT a CrewAI agent.** It's a lightweight Python class that:
- Runs on the main async event loop (immediate responses)
- Always reads from the database to see what other agents have done
- Can influence story (store preferences) but cannot directly instruct other agents
- Emits events that the Coordinator may pick up

**Other agents have no "front":**
- StructureAgent, CharacterAgent, NarrativeAgent, etc. are all CrewAI agents
- They run via `crew.kickoff()` which is synchronous and blocking
- Only the Coordinator can orchestrate them (no direct API endpoints)
- They run in a dedicated thread pool to avoid blocking the user

### Services (`src/services/`)

**Storage:**
| Service | File | Purpose |
|---------|------|---------|
| DatabaseService | `database.py` | Azure SQL CRUD for households, stories, chapters, characters |
| BlobStorageService | `blob_storage.py` | Audio file storage with SAS URL generation |
| FirebaseService | `firebase.py` | Legacy Firebase support (`USE_AZURE_SQL=false`) |

**LLM:**
| Service | File | Purpose |
|---------|------|---------|
| LLMRouter | `llm_router.py` | Centralized model selection with hierarchy fallback to `azure/model-router` |
| FoundryService | `foundry.py` | Azure AI Foundry Model Router client |
| AdaptiveRateLimiter | `adaptive_rate_limiter.py` | Per-provider concurrency with priority lanes for dialogue |

**Voice:**
| Service | File | Purpose |
|---------|------|---------|
| VoiceService | `voice.py` | Multi-provider TTS (ElevenLabs, Speechify, OpenAI) |
| SSMLProcessor | `ssml_processor.py` | SSML markup generation (kept for provider flexibility) |

**Utilities:**
| Service | File | Purpose |
|---------|------|---------|
| EventEmitter | `events.py` | Real-time WebSocket event broadcasting |
| CharacterService | `character_service.py` | Norse-aware character deduplication and matching |
| LoreLanternLogger | `logger.py` | Two-mode logging (terminal + JSON debug files) |
| InputClassifier | `input_classifier.py` | User input tier classification (immediate vs. story influence) |

---

## Multi-Provider LLM Architecture

Lore Lantern uses **LiteLLM** as a universal adapter to communicate with any LLM provider through a consistent interface. This enables mixing providers per-agent for optimal cost/quality tradeoffs.

### Supported Providers

| Provider | Prefix | Example Models | Use Case |
|----------|--------|----------------|----------|
| **Anthropic** | `claude-` | `claude-opus-4-5-20251101`, `claude-sonnet-4-5-20250929` | Creative writing, character depth |
| **OpenAI** | `gpt-` | `gpt-5.2`, `gpt-4o-mini` | Fast fact-checking, dialogue |
| **Google** | `gemini/` | `gemini-3-pro-preview`, `gemini-3-flash-preview` | Structure planning, voice direction |
| **Azure Foundry** | `azure/` | `azure/model-router` | Auto-routing with cost optimization |
| **Ollama (Local)** | `ollama/` | `ollama/llama3.3:70b`, `ollama/borealis-4b` | Local inference, language refinement |

### How LiteLLM Works

```python
# All providers use the same interface
from litellm import completion

# Anthropic
response = completion(model="claude-opus-4-5-20251101", messages=[...])

# OpenAI
response = completion(model="gpt-5.2", messages=[...])

# Google
response = completion(model="gemini/gemini-3-pro-preview", messages=[...])

# Local Ollama
response = completion(model="ollama/llama3.3:70b", messages=[...], api_base="http://localhost:11434")
```

### Provider Selection Strategy

Based on extensive E2E testing (see [LLM Configuration Analysis Report](docs/LLM_CONFIGURATION_ANALYSIS_REPORT.md)), we found:

| Task Type | Recommended Provider | Why |
|-----------|---------------------|-----|
| **Creative Writing** | Anthropic (Opus) | Best constraint adherence, age-appropriate language |
| **Fact Checking** | OpenAI (GPT-5.2) | Fast, accurate, good reasoning |
| **Structure Planning** | Google (Gemini Pro) | 2x faster, excellent at outlining |
| **Voice Direction** | Google (Gemini Flash) | Fast markup, good instruction following |
| **Dialogue** | OpenAI (GPT-4o-mini) | Sub-second latency, conversational |
| **Language Refinement** | Ollama (Borealis 4B) | Local, Norwegian-specialized |

---

## LLM Router Configuration

Model configuration is in `src/config/models.yaml`. The LLM Router resolves models with this hierarchy:

```
1. TEST_<AGENT>_MODEL env var   (e.g., TEST_NARRATIVE_MODEL)
2. TEST_<GROUP>_MODEL env var   (e.g., TEST_ROUNDTABLE_MODEL)
3. Agent-specific model in models.yaml
4. Group model in models.yaml
5. Default: azure/model-router
```

**Environment variable overrides for A/B testing:**
| Variable | Example | Effect |
|----------|---------|--------|
| `TEST_NARRATIVE_MODEL` | `gpt-5.2` | Override single agent |
| `TEST_ROUNDTABLE_MODEL` | `gemini-3-flash-preview` | Override all 6 reviewers |

If nothing is configured, the system falls back to `azure/model-router`.

---

## Local Model Support (Ollama)

Set `OLLAMA_BASE_URL` in `.env` to use local models. See `.env.example` for details.

The `language_refiner` agent uses NB AI Lab's Borealis model locally to polish Norwegian text.

---

## Model Testing & Selection

We conducted extensive E2E testing to identify optimal models. See the full analysis: [`docs/LLM_CONFIGURATION_ANALYSIS_REPORT.md`](docs/LLM_CONFIGURATION_ANALYSIS_REPORT.md)

**Key Finding:** OpenAI models ignore word count limits (+571% over target). Anthropic provides best constraint adherence (+22%).

**Our optimized configuration:**
- **Creative tasks** (narrative, characters): Anthropic Opus - best constraint adherence
- **Fast tasks** (factcheck, dialogue): OpenAI - speed where quality threshold is met
- **Planning tasks** (structure, voice): Google Gemini - 2x faster than Opus
- **Local refinement**: Ollama Borealis - free, Norwegian-specialized

---

## Azure AI Foundry Model Router

The Model Router automatically selects the optimal model based on prompt complexity, balancing quality and cost. However, we use the `reasoning_effort` parameter to influence model selection toward higher-quality models.

### Why We Use `reasoning_effort`

By default, the Model Router often selects GPT-4o-mini for cost optimization. For story generation tasks that require creativity and nuance, we want higher-quality models like:

- **Claude Sonnet 4.5** - Excellent for narrative writing
- **GPT-5** - Strong reasoning and creativity
- **GPT-OSS-120B** - Open-source alternative with good quality

Setting `reasoning_effort` hints to the router that we need a more capable model:

```python
# In narrative.py and structure.py
llm_kwargs["extra_body"] = {"reasoning_effort": "medium"}
```

### `reasoning_effort` Values

| Value | Effect | Use Case |
|-------|--------|----------|
| `"low"` | Fast, cheaper models (GPT-4o-mini) | Dialogue, quick responses |
| `"medium"` | Balanced quality/speed | **Story generation, chapter writing** |
| `"high"` | Maximum quality models | Complex reasoning tasks |

### Known Issue: `"high"` + GPT-5-nano

When `reasoning_effort: "high"` is set, the router may select `gpt-5-nano-2025-08-07`. This model has a **critical issue**: it can spend ALL available tokens on internal reasoning, leaving 0 tokens for actual content output.

**Symptoms:**
- `finish_reason: "length"` with `content: ""`
- `reasoning_tokens: 16384` (all tokens used for thinking)
- Empty responses causing retries and failures

**Solution:** Use `"medium"` instead of `"high"` for content generation tasks. This still selects quality models while ensuring tokens are reserved for output.

---

## Data Flow & Azure Integration

Lore Lantern uses Azure SQL Database for structured data and Azure Blob Storage for audio files, with a clean separation between the coordinator (orchestration) and services (persistence).

### End-to-End Story Generation Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLIENT REQUEST                                   │
│  POST /api/stories/init                                                 │
│  { prompt: "Tell me about Vikings", child_id: "..." }                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI ROUTES                                   │
│  src/api/routes.py                                                      │
│  - Validates input                                                       │
│  - Creates Story object                                                  │
│  - Saves to Azure SQL                                                    │
│  - Returns story_id                                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       STORY COORDINATOR                                  │
│  src/crew/coordinator.py                                                │
│                                                                          │
│  1. StructureAgent    → Create 5-chapter outline                        │
│  2. CharacterAgent    → Generate character profiles                     │
│  3. For each chapter:                                                    │
│     a. NarrativeAgent  → Write chapter draft                            │
│     b. Round Table     → 6 parallel reviewers                           │
│     c. VoiceDirector   → Add audio tags                                 │
│     d. TTS Service     → Generate audio → Blob Storage                  │
│  4. Save progress to Azure SQL after each step                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌──────────────────────────────┐    ┌──────────────────────────────┐
│       AZURE SQL DATABASE      │    │     AZURE BLOB STORAGE        │
│                               │    │                               │
│  households                   │    │  lorelantern-audio/           │
│    └─ children               │    │    └─ {household_id}/          │
│         └─ stories           │    │         └─ {story_id}/         │
│              ├─ chapters     │    │              ├─ chapter_1.mp3  │
│              ├─ characters   │    │              ├─ chapter_2.mp3  │
│              └─ dialogues    │    │              └─ ...            │
└──────────────────────────────┘    └──────────────────────────────┘
```

### Database Schema (Azure SQL)

The relational structure enforces household isolation and data integrity:

```sql
households (id, display_name, language)
    │
    ├── children (id, household_id, name, birth_year, active_story_id)
    │       │
    │       ├── stories (id, household_id, child_id, prompt, status, structure, preferences)
    │       │       │
    │       │       ├── chapters (id, story_id, number, title, content, tts_content, audio_blob_url, status)
    │       │       │
    │       │       ├── characters (id, story_id, name, role, personality_traits, character_arc)
    │       │       │
    │       │       └── dialogues (id, story_id, speaker, message, timestamp)
    │       │
    │       └── learning_progress (child_id, vocabulary, concepts, reading_level)
    │
    └── reading_states (story_id, current_chapter, chapter_position, playback_phase)
```

### Key Database Operations

| Operation | Service Method | Description |
|-----------|---------------|-------------|
| Create story | `db.save_story(story)` | Inserts story with household isolation |
| Save chapter | `db.save_chapter(story_id, chapter)` | Saves chapter content + metadata |
| Get story | `db.get_story(story_id)` | Returns full story with chapters, characters |
| Update status | `db.update_story_status(story_id, status)` | Updates generation progress |
| Save dialogue | `db.save_dialogue(story_id, dialogue)` | Persists conversation history |

### Blob Storage Structure

Audio files are organized by household for multi-tenant isolation:

```
lorelantern-audio/
├── {household_id}/
│   ├── {story_id}/
│   │   ├── chapter_1.mp3              # Full chapter audio
│   │   ├── chapter_1_segment_0.mp3    # Optional chunked audio
│   │   ├── chapter_1_segment_1.mp3
│   │   ├── chapter_2.mp3
│   │   └── ...
│   └── {another_story_id}/
│       └── ...
└── {another_household_id}/
    └── ...
```

### Service Layer Architecture

```python
# src/services/__init__.py exposes all services

DatabaseService      # Azure SQL operations (CRUD for all entities)
BlobStorageService   # Azure Blob operations (audio upload/download)
VoiceService         # TTS generation (Speechify/ElevenLabs/OpenAI)
LLMRouter            # Model selection and LiteLLM calls
```

### How the Coordinator Uses Services

The `StoryCoordinator` (`src/crew/coordinator.py`) orchestrates the entire pipeline:

```python
# Simplified flow from coordinator.py

async def generate_story(self, story_id: str):
    # 1. Load story from database
    story = await self.db.get_story(story_id)

    # 2. Generate structure (StructureAgent)
    structure = await self.structure_agent.create_outline(story.prompt)
    story.structure = structure
    await self.db.save_story(story)  # Persist immediately

    # 3. Generate characters (CharacterAgent)
    characters = await self.character_agent.create_characters(structure)
    for char in characters:
        await self.db.save_character(story_id, char)

    # 4. Generate each chapter
    for chapter_num in range(1, 6):
        # Generate chapter content
        chapter = await self.narrative_agent.write_chapter(...)

        # Round Table review
        reviews = await self._run_round_table(chapter)
        chapter.round_table_review = reviews

        # Voice direction
        chapter.tts_content = await self.voice_director.add_tags(chapter.content)

        # Save chapter to database
        await self.db.save_chapter(story_id, chapter)

        # Generate and upload audio
        audio_bytes = await self.voice_service.generate_audio(chapter.tts_content)
        household_id = await self.db.get_household_id_for_story(story_id)
        audio_url = await self.blob.upload_audio(household_id, story_id, chapter_num, audio_bytes)

        # Update chapter with audio URL
        chapter.audio_blob_url = audio_url
        await self.db.save_chapter(story_id, chapter)
```

### UUID Handling

Azure SQL requires valid UUIDs. The `DatabaseService` automatically converts string IDs:

```python
# database.py
def to_uuid(value: str) -> str:
    if is_valid_uuid(value):
        return str(UUID(value))
    # Generate deterministic UUID5 from string
    return str(uuid.uuid5(LORELANTERN_NAMESPACE, value))
```

This means legacy IDs like `"parent_123"` become deterministic UUIDs, ensuring the same input always produces the same UUID.

### Real-Time Updates (WebSocket)

During generation, the coordinator broadcasts progress via WebSocket:

```python
# Broadcast chapter completion
await self.ws_manager.broadcast(story_id, {
    "type": "chapter_complete",
    "chapter": chapter_num,
    "title": chapter.title,
    "audio_url": chapter.audio_blob_url
})
```

Clients connect to `/ws/story/{story_id}` to receive live updates.

---

## AI Agents

Lore Lantern uses 10 specialized AI agents, each with a distinct persona and optimal model assignment.

### Story Generation Agents

| Agent | Persona | Model | Role |
|-------|---------|-------|------|
| **StructureAgent** | Guillermo del Toro | `gemini-3-pro-preview` | Story architecture, chapter outlines |
| **CharacterAgent** | Clarissa Pinkola Estés | `claude-opus-4-5-20251101` | Jungian character development, D&D progression |
| **NarrativeAgent** | Nnedi Okofor (Africanfuturist) | `claude-opus-4-5-20251101` | Chapter writing, prose craft |
| **VoiceDirectorAgent** | Jim Dale | `gemini-3-flash-preview` | SSML/audio tags for expressive narration |
| **CompanionAgent** | Hanan (Friendly Teacher) | `gpt-4o-mini` | Real-time dialogue, child interaction |
| **LanguageRefiner** | Borealis | `ollama/borealis-4b` | Norwegian text refinement (local) |

### Round Table Reviewers (6 Parallel)

After each chapter draft, 6 reviewers analyze simultaneously:

| Reviewer | Persona | Model | Focus | Verdict Options |
|----------|---------|-------|-------|-----------------|
| **Guillermo** | del Toro | `gemini-3-pro-preview` | Structure, pacing, visual themes | approve / concern / block |
| **Bill** | Bill Bryson | `gpt-5.2` | Historical/scientific accuracy | approve / concern / block |
| **Clarissa** | Pinkola Estés | `claude-opus-4-5-20251101` | Character psychology, voice | approve / concern / block |
| **Benjamin** | Benjamin Dreyer | `claude-sonnet-4-5-20250929` | Prose quality, rhythm, humor | approve / concern / block |
| **Continuity** | Editor | `claude-sonnet-4-5-20250929` | Plot threads, setup→payoff | approve / concern / block |
| **Stephen** | Stephen King | `claude-sonnet-4-5-20250929` | Tension, hooks, chapter endings | approve / concern / block |

### Agent Groups

Agents are organized into groups for bulk configuration:

```yaml
groups:
  roundtable: [structure, factcheck, character, line_editor, continuity, tension]
  writers: [narrative, dialogue]
  post: [voice_director]
  refiners: [language_refiner]
```

Override an entire group:
```bash
TEST_ROUNDTABLE_MODEL="gemini-3-flash-preview" python -m src.main
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-repo/lorelantern.git
cd lorelantern
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your credentials (see .env.example for all options)

# 3. Run the server
python -m src.main

# 4. Verify with E2E tests (see Test-Driven Development section)
python tests/e2e_inger_helene_norwegian.py
```

---

## API Endpoints

### Stories

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/stories/init` | Initialize a new story |
| GET | `/api/stories/{id}` | Get story details |
| POST | `/api/stories/{id}/generate-all` | Generate all chapters |
| GET | `/api/stories/{id}/chapters/{num}` | Get specific chapter |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `/ws/story/{id}` | Real-time story updates |
| `/ws/companion/{id}` | Dialogue with CompanionAgent |

### Parent Dashboard

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/parents` | Create parent account |
| POST | `/api/parents/{id}/children` | Add child profile |
| GET | `/api/parents/{id}/stories` | List child's stories |

### Input Validation Limits

All user input has consistent character limits to support long story prompts with reference material:

| Field | Max Length | Description |
|-------|------------|-------------|
| `prompt` | 20,000 chars | Story prompt including reference material, articles |
| `message` | 20,000 chars | Conversation/dialogue messages |
| `pre_story_messages` | 20,000 chars each | Pre-story conversation context |

**Use Case:** Users can paste article excerpts, book passages, or detailed world-building notes (like the Mass Effect example) directly into story prompts.

---

## Project Structure

```
lorelantern/
├── src/
│   ├── agents/                 # AI agent definitions
│   ├── api/                    # REST + WebSocket endpoints
│   ├── crew/                   # Story orchestration, Round Table
│   ├── models/                 # Pydantic data models
│   ├── services/               # Database, voice, logging
│   ├── prompts/                # Agent system prompts
│   └── main.py                 # FastAPI application
├── tests/                      # E2E and unit tests
├── static/                     # Frontend assets (Debug Studio)
└── [root files documented below]
```

---

## Root Files Reference

Every file in the project root has a specific purpose:

### Configuration Files

| File | Purpose | Git Tracked |
|------|---------|-------------|
| `.env` | Environment variables (API keys, database credentials) | ❌ No (secrets) |
| `.env.example` | Template for `.env` - copy and fill in your values | ✅ Yes |
| `.gitignore` | Files excluded from git | ✅ Yes |
| `.deployment` | Azure subscription/resource IDs for deployment scripts | ❌ No (sensitive) |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | Main project documentation (this file) |
| `ARCHITECTURE_RULES.md` | Coding standards, data flow, agent patterns |
| `AGENT_ARCHITECTURE.md` | Detailed agent system documentation |
| `WEBSOCKET_VOICE_SYSTEM.md` | Real-time voice/WebSocket architecture |
| `LICENSE` | AGPL-3.0 license |

### Deployment & Dependencies

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies for **Azure/production** (includes `pysqlite3-binary`) |
| `requirements_mac.txt` | Python dependencies for **macOS/local dev** (no pysqlite3-binary) |
| `startup.sh` | Azure App Service startup script (gunicorn + uvicorn) |

### Development-Only Files (Not for Production)

| File | Purpose | Git Tracked |
|------|---------|-------------|
| `firebase-credentials.json` | Google Cloud/Firebase service account (legacy, dev-only) | ❌ No |

**Why two requirements files?**
- `pysqlite3-binary` is needed on Azure App Service (Linux) to fix SQLite version issues
- On macOS, the system SQLite works fine, and `pysqlite3-binary` fails to install
- Use `requirements_mac.txt` for local development, `requirements.txt` for deployment

---

## Scripts (Developer Toolbox)

The `scripts/` folder contains utility scripts for development and debugging.
All scripts load configuration from `.env` file.

### Database & Storage Scripts

| Script | Purpose | .env Variables |
|--------|---------|----------------|
| `clean_slate_azure.py` | **Wipe database** - Deletes all data from Azure SQL + Blob Storage | `AZURE_SQL_*`, `AZURE_BLOB_*` |
| `schema.sql` | **SQL schema** - Table definitions (run in Azure Portal Query Editor) | N/A |
| `test_azure_sql.py` | **Connection test** - Verifies Azure SQL connectivity and CRUD operations | `AZURE_SQL_*` |

```bash
# Dry run (shows what would be deleted)
python scripts/clean_slate_azure.py --dry-run

# Actually delete everything
python scripts/clean_slate_azure.py --confirm
```

### API Validation Scripts

| Script | Purpose | .env Variables |
|--------|---------|----------------|
| `validate_api_keys.py` | **Test all APIs** - Makes real API calls to verify keys work | All API keys |
| `test_model_router_selection.py` | **Model router analysis** - Tests which models Foundry selects | `FOUNDRY_*` |

```bash
# Test all providers (LLM + TTS)
python scripts/validate_api_keys.py

# Test only LLM providers
python scripts/validate_api_keys.py --llm-only

# Test only TTS providers
python scripts/validate_api_keys.py --tts-only

# Test specific provider
python scripts/validate_api_keys.py --provider foundry
python scripts/validate_api_keys.py --provider elevenlabs
```

### TTS Voice Explorer

| Script | Purpose | .env Variables |
|--------|---------|----------------|
| `list_tts_voices.py` | **Explore voices** - Lists all available TTS voices by provider/language | All TTS API keys |

```bash
# List all providers
python scripts/list_tts_voices.py

# Filter by provider
python scripts/list_tts_voices.py --provider elevenlabs
python scripts/list_tts_voices.py --provider speechify

# Filter by language
python scripts/list_tts_voices.py --language no
```

### Code Quality Scripts

| Script | Purpose | .env Variables |
|--------|---------|----------------|
| `validate_consistency.py` | **Architecture linter** - Checks code against ARCHITECTURE_RULES.md | N/A |

```bash
python scripts/validate_consistency.py
```

**Script outputs** are saved to `scripts/outputs/` (gitignored).

---

## Tests

The `tests/` folder contains E2E and unit tests. All tests require the server to be running (`python -m src.main`).

### E2E Tests (End-to-End)

| Test | Language | Child | Focus |
|------|----------|-------|-------|
| `e2e_inger_helene_norwegian.py` | Norwegian | Inger Helene (5) | 3-step conversation flow, A/B testing |
| `e2e_harald_emma.py` | English | Emma (7) | Viking story, prefetch, non-blocking dialogue |

**What E2E tests validate:**
- ✅ Character completeness (5/5 characters created)
- ✅ Chapter completeness (chapters generated)
- ✅ Language content (Norwegian/English detected)
- ✅ Dialogue grounding (story-specific responses)
- ✅ Non-blocking dialogue (responses during generation)
- ✅ Chapter prefetch behavior

```bash
# Start server first
python -m src.main &

# Run Norwegian test
python tests/e2e_inger_helene_norwegian.py

# Run English test
python tests/e2e_harald_emma.py

# A/B test with different models
TEST_NARRATIVE_MODEL=gpt-4o python tests/e2e_inger_helene_norwegian.py
TEST_NARRATIVE_MODEL=claude-sonnet-4-5-20250929 python tests/e2e_inger_helene_norwegian.py
```

### Unit Tests

| Test | Purpose |
|------|---------|
| `test_intent_classifier.py` | Pattern matching for conversation routing |

```bash
python tests/test_intent_classifier.py
```

**Test outputs** (generated stories as markdown) are saved to `tests/outputs/` (gitignored).

**Daily quality tracking:** Run E2E tests daily to track quality variations across model updates.

---

## Key Features

### Character Evolution

Characters grow across chapters. The NarrativeAgent receives:
- Previous chapter content (800 chars per chapter)
- Character relationship changes
- Personality evolution notes
- Skills learned
- Current emotional state

### Sequential Chapter Dependencies

Chapters must be generated in order. The system:
- Verifies Chapter N-1 exists before writing Chapter N
- Retries up to 3 times with 5-second delays
- Stops generation if a chapter fails (can't skip)

### Round Table Quality Assurance

Each chapter goes through parallel review:
1. **Draft** → NarrativeAgent writes chapter
2. **Review** → 5 agents review simultaneously
3. **Revise** → NarrativeAgent addresses feedback
4. **Approve** → All reviewers must approve or have only "concerns"

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.11, FastAPI, asyncio |
| **AI Orchestration** | CrewAI, LiteLLM |
| **Models** | Azure AI Foundry (GPT-5, Claude Haiku) |
| **Database** | Azure SQL Database |
| **Storage** | Azure Blob Storage |
| **TTS** | OpenAI TTS, Speechify, ElevenLabs |
| **Validation** | Pydantic v2 |

---

## Architecture Decision: Firebase → Azure SQL

### Why We Migrated

The project started with **Firebase Realtime Database** but migrated to **Azure SQL Database**. Here's what we learned:

| Aspect | Firebase | Azure SQL | Winner |
|--------|----------|-----------|--------|
| **Data Model** | NoSQL (JSON tree) | Relational (tables) | Azure SQL |
| **Queries** | Limited (no JOINs) | Full SQL | Azure SQL |
| **Household Isolation** | Manual path structure | Foreign keys | Azure SQL |
| **Cost at Scale** | Pay per operation | Predictable | Azure SQL |
| **Azure Integration** | Separate ecosystem | Native (Blob, AI) | Azure SQL |
| **Offline Sync** | Built-in | Not needed | - |

### Key Lessons Learned

1. **NoSQL Friction for Relational Data**
   - Stories have chapters, chapters have characters - relational by nature
   - Firebase path structure (`/stories/{id}/chapters/{n}`) became unwieldy
   - No way to query "all chapters where character X appears"

2. **Household Isolation**
   - Firebase required careful path-based access control
   - Azure SQL uses proper foreign keys: `household_id → stories → chapters`
   - Much cleaner data integrity

3. **Azure Ecosystem Benefits**
   - **Blob Storage** for audio (same subscription)
   - **AI Foundry** for LLMs (same subscription)
   - **App Service** for hosting (same subscription)
   - Unified billing, networking, and identity

4. **Migration Strategy**
   - Created `DatabaseService` with same interface as `FirebaseService`
   - Feature flag: `USE_AZURE_SQL=true` switches between them
   - Both services implement same methods (`save_story`, `get_story`, etc.)
   - Old Firebase code kept for reference but not used

### File Structure

```
src/services/
├── database.py     ← Azure SQL (active when USE_AZURE_SQL=true)
├── firebase.py     ← Firebase (legacy, kept for reference)
└── blob_storage.py ← Azure Blob (audio files)
```

### When to Use Each

| Use Case | Recommendation |
|----------|----------------|
| **New deployments** | Azure SQL (`USE_AZURE_SQL=true`) |
| **Existing Firebase users** | Can still use Firebase (`USE_AZURE_SQL=false`) |
| **Local development** | Azure SQL (matches production) |
| **Offline-first mobile** | Consider Firebase (has offline sync) |

---

## Architecture Decision: TTS Provider Strategy

### Three Approaches Tested

We evaluated three different approaches to expressive narration:

| Approach | Implementation | Pros | Cons |
|----------|---------------|------|------|
| **SSML (W3C Standard)** | `SSMLProcessor` | Provider-agnostic, industry standard | Limited expressiveness, verbose |
| **ElevenLabs Audio Tags** | `VoiceDirectorAgent` | Highly expressive (`[whispers]`, `[laughing]`), natural | Proprietary, ElevenLabs-only |
| **Speechify** | Direct API | Good Norwegian support, reliable | Less expressive than ElevenLabs |

### Current Choice: ElevenLabs Audio Tags

We chose ElevenLabs with proprietary audio tags because:
1. **Expressiveness** - Tags like `[whispers]`, `[laughing]`, `[excited]` create engaging narration
2. **VoiceDirectorAgent** - Jim Dale-inspired agent adds dramatic timing and emotion
3. **Quality** - eleven_v3 model produces the most natural-sounding narration
4. **Multilingual** - Explicit `language_code` ensures Norwegian sounds Norwegian

### SSMLProcessor: Kept for Future Flexibility

The `SSMLProcessor` (`src/services/ssml_processor.py`) is **intentionally kept** even though we use ElevenLabs:

1. **Provider-Agnostic** - Works with any SSML-compatible TTS (Google, Azure, Amazon)
2. **Fallback Option** - If ElevenLabs pricing/quality changes, we can switch
3. **Industry Standard** - W3C SSML is widely supported
4. **No Vendor Lock-in** - Reduces dependency on single provider

### File Structure

```
src/services/
├── voice.py           ← TTS orchestration (provider selection, fallbacks)
├── ssml_processor.py  ← Provider-agnostic SSML (kept for future use)
└── ...

src/agents/
└── voice_director.py  ← Adds ElevenLabs audio tags to prose
```

### Provider Priority (Current)

```
Narration: ElevenLabs → Speechify → OpenAI tts-1
Dialogue:  OpenAI gpt-4o-mini-tts (or Direct Audio)
```

---

## License

**GNU Affero General Public License v3.0 (AGPL-3.0)**

- Free to use, modify, and distribute
- If you run this as a service, you must share your changes
- Derivative works must also be AGPL-3.0

For commercial licensing (closed-source usage), please open an issue.

---

*Last updated: January 2026*
*E2E tested with Norwegian language stories*
