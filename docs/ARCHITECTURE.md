# Lore Lantern - Full Developer Guide

Complete technical documentation for the multi-agent AI storytelling system.

**Version:** 2.0.0
**Last Updated:** December 2024
**Files:** 64 Python files across `/src/`

---

## Table of Contents

1. [Overview & Pipeline](#1-overview--pipeline)
2. [Agent System](#2-agent-system)
3. [Services Layer](#3-services-layer)
4. [API Reference](#4-api-reference)
5. [Data Models](#5-data-models)
6. [Prompt System](#6-prompt-system)
7. [Configuration](#7-configuration)

---

# 1. Overview & Pipeline

## System Overview

Lore Lantern is an AI-powered storytelling system that generates personalized children's stories through a collaborative multi-agent architecture. The system features:

- **11 specialized AI agents** with distinct personas and expertise
- **Round Table review** where 6 agents collaboratively review each chapter
- **D&D-style character progression** with skills that evolve through the story
- **Multi-language support** (English, Norwegian, Spanish)
- **Voice synthesis** via ElevenLabs, Speechify, and OpenAI
- **Hybrid generation** allowing user input during story playback

## Story Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     STORY GENERATION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. initialize_story()                                               │
│     └─> Creates Story object                                         │
│     └─> Status: IN_DIALOGUE                                          │
│                                                                      │
│  2. generate_story_structure()                                       │
│     └─> StructureAgent (Guillermo) creates 3-7 chapter outline       │
│     └─> Narrative Method Debate (Guillermo + Stephen decide POV)     │
│     └─> Status: STRUCTURE_READY                                      │
│                                                                      │
│  3. create_characters()                                              │
│     └─> CharacterAgent (Clarissa) creates each character SEQUENTIALLY│
│     └─> D&D-style skills with levels 1-10                            │
│     └─> Deduplication: rule-based + LLM (Bill) verification          │
│     └─> Status: CHARACTERS_READY                                     │
│                                                                      │
│  4. refine_structure() [Post Chapter 1]                              │
│     └─> Updates synopses based on actual Chapter 1 content           │
│     └─> Leverages character skills in future chapter plans           │
│                                                                      │
│  5. For each chapter (1 to N):                                       │
│                                                                      │
│     5a. write_chapter()                                              │
│         └─> NarrativeAgent (Nnedi) writes draft                      │
│         └─> Uses "Story So Far" context                              │
│         └─> Incorporates queued user inputs                          │
│                                                                      │
│     5b. round_table_review()                                         │
│         └─> 6 agents review in parallel (max 2 concurrent)           │
│         └─> Verdicts: APPROVE / CONCERN / BLOCK                      │
│         └─> Decision: BLOCK → revise, CONCERN → polish               │
│                                                                      │
│     5c. evolve_characters_post_chapter()                             │
│         └─> CharacterAgent updates progression                       │
│         └─> Skills gained, personality evolved, relationships changed│
│                                                                      │
│     5d. apply_voice_direction()                                      │
│         └─> VoiceDirectorAgent (Jim Dale) adds [audio tags]          │
│                                                                      │
│     5e. generate_audio()                                             │
│         └─> TTS via ElevenLabs/Speechify/OpenAI                      │
│         └─> Cached to Azure Blob Storage                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Status Flow

```
INITIALIZING → IN_DIALOGUE → STRUCTURE_READY → CHARACTERS_READY → GENERATING_CHAPTER → COMPLETED
                                                                         ↓
                                                                      PAUSED
                                                                         ↓
                                                                      FAILED
```

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/crew/coordinator.py` | ~3,500 | Main orchestration, all pipeline methods |
| `src/api/routes.py` | ~2,000 | REST API endpoints |
| `src/api/websocket.py` | ~1,400 | Real-time WebSocket handlers |
| `src/services/voice.py` | ~1,400 | Multi-provider TTS |
| `src/main.py` | ~430 | Application startup |

---

# 2. Agent System

## Agent Roster (11 Agents)

| Agent | Persona | Temperature | Role |
|-------|---------|-------------|------|
| **StructureAgent** | Guillermo del Toro | 0.7 | Story architect - creates 3-7 chapter outlines with educational goals, world-building notes |
| **CharacterAgent** | Dr. Clarissa Pinkola Estés | 0.7 | Jungian psychologist - creates characters with psychological depth, D&D skills, arc milestones |
| **NarrativeAgent** | Africanfuturist writer (Nnedi Okofor) | 0.85 | Prose author - writes chapter content with sensory grounding and immersion |
| **FactCheckAgent** | Bill Nye | 0.25 | Science educator - verifies historical/scientific accuracy, detects duplicates |
| **ContinuityAgent** | Continuity Editor | 0.3 | Plot thread tracker - tracks mysteries, objects, promises, ensures setup → payoff |
| **LineEditorAgent** | Benjamin Dreyer | 0.4 | Copy chief - polishes prose rhythm, show-don't-tell, read-aloud appeal |
| **TensionAgent** | Stephen King | 0.4 | Page-turner architect - ensures hooks, momentum, "turn the page" test |
| **VoiceDirectorAgent** | Jim Dale | 0.55 | Audiobook narrator - adds [audio tags] for expressive TTS |
| **DialogueAgent** | Enthusiastic Teacher | 0.7 | User-facing chatbot for story setup conversations |
| **CompanionAgent** | Hanan al-Hroub | N/A | Lightweight engagement agent - fills wait times with teasers (not CrewAI) |

## Round Table Review

After each chapter draft, 6 agents review simultaneously (max 2 concurrent via ThreadPoolExecutor):

```
┌─────────────────────────────────────────────────────────────────────┐
│                       ROUND TABLE REVIEW                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  GUILLERMO (Structure)     "Does chapter fit the story arc?"         │
│  ├─ Evaluates: 4-part structure, pacing, thematic consistency        │
│  └─ BLOCKS if: No climax, abrupt ending, >50% exposition             │
│                                                                      │
│  BILL (Facts)              "Is content historically accurate?"       │
│  ├─ Evaluates: Historical accuracy, scientific plausibility, age-fit │
│  └─ BLOCKS if: Creates false understanding, dangerous misinformation │
│                                                                      │
│  CLARISSA (Characters)     "Are characters psychologically true?"    │
│  ├─ Evaluates: Voice consistency, arc progression, shadow/growth     │
│  └─ BLOCKS if: Character acts against personality without reason     │
│                                                                      │
│  BENJAMIN (Prose)          "Is the writing polished?"                │
│  ├─ Evaluates: Sentence rhythm, show-don't-tell, sensory grounding   │
│  └─ BLOCKS if: sensory_score = "needs_work", POV breaks              │
│                                                                      │
│  CONTINUITY                "Are plot threads tracked?"               │
│  ├─ Evaluates: New elements, resolutions, dangling threads           │
│  └─ BLOCKS if: Final chapter with unresolved major thread            │
│                                                                      │
│  STEPHEN (Tension)         "Does chapter end with a hook?"           │
│  ├─ Evaluates: Chapter ending, tension architecture, pacing          │
│  └─ BLOCKS if: Reader can put book down satisfied (except finale)    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │          DECISION             │
                    ├───────────────────────────────┤
                    │  Any BLOCK?    → Revise (max 3)│
                    │  Only CONCERN? → Polish pass   │
                    │  All APPROVE?  → Finalize      │
                    └───────────────────────────────┘
```

## Verdict System

| Verdict | Meaning | Action |
|---------|---------|--------|
| **APPROVE** | Meets all criteria | Proceed to finalize |
| **CONCERN** | Issues but fixable | Apply polish pass (subtle refinements) |
| **BLOCK** | Major issue | Mandatory revision (NarrativeAgent rewrites) |

## Character Progression (D&D-Style)

Characters evolve after each chapter:

| Attribute | Description | Example |
|-----------|-------------|---------|
| `skills_learned` | New skills with level 1-10, acquired_chapter | `{"name": "Leadership", "level": 3, "acquired_chapter": 2}` |
| `skills_improved` | Existing skills leveled up | Level 2 → Level 4 |
| `personality_evolution` | Trait changes with trigger events | `{"from": "timid", "to": "brave", "trigger": "saved friend"}` |
| `relationship_changes` | New bonds, conflicts | `{"other": "Jarl Erik", "type": "ally", "strength": 7}` |
| `current_emotional_state` | Never "neutral" | "determined", "anxious", "hopeful" |

## ContinuityAgent Tracking

Monitors 6 plot element types:

| Type | Description | Resolution Required |
|------|-------------|---------------------|
| **MYSTERY** | Questions needing answers | Must reveal answer |
| **OBJECT** | Items with narrative importance | Must show significance |
| **CONFLICT** | Tensions needing resolution | Must resolve or explain |
| **PROMISE** | Character commitments | Must keep or meaningfully break |
| **RELATIONSHIP** | Bonds needing development | Must develop or explain stasis |
| **SECRET** | Hidden information | Must reveal before end |

Each tracked as: `introduced` → `developed` → `resolved` or `at_risk`

---

# 3. Services Layer

## Storage Services

### FirebaseService (`firebase.py`)
**Purpose:** Legacy Firebase Realtime Database backend (deprecated)

| Method | Description |
|--------|-------------|
| `save_story()` | Persist story object |
| `get_story()` | Retrieve by ID |
| `save_chapter()` | Save chapter content |
| `get_characters()` | Get all characters for story |
| `queue_user_input()` | Queue input for hybrid generation |
| `shutdown()` | Clean up ThreadPoolExecutor |

### DatabaseService (`database.py`)
**Purpose:** Azure SQL Database backend (new)

- Replaces Firebase for structured data
- UUID conversion for Azure SQL compatibility
- JSON serialization for complex types
- Household-level data isolation (multi-tenancy)

### BlobStorageService (`blob_storage.py`)
**Purpose:** Azure Blob Storage for audio files

| Method | Description |
|--------|-------------|
| `upload_audio()` | Upload MP3 to blob |
| `get_audio_url()` | Generate 24h SAS URL |
| `delete_story_audio()` | Clean up story audio |
| `list_chapter_segments()` | List all segment URLs |

**Path Pattern:** `{household_id}/{story_id}/chapter_{num}[_segment_{n}].mp3`

## AI/LLM Services

### FoundryService (`foundry.py`)
**Purpose:** Microsoft Azure AI Foundry Model Router

- Automatically routes to optimal model (GPT, Claude, DeepSeek, Llama, Grok)
- Three routing modes: `quality`, `cost`, `balanced`
- Tracks actual model selected in responses

| Method | Description |
|--------|-------------|
| `chat_completion()` | Async LLM request |
| `chat_completion_sync()` | Sync for CrewAI |
| `classify_input()` | Fast classification via balanced mode |

### ModelTracker (`model_tracker.py`)
**Purpose:** LiteLLM callback for tracking actual model used

- Registers LiteLLM success callback at startup
- Thread-safe singleton
- Enables observatory events with real model names

### ModelValidator (`model_validator.py`)
**Purpose:** Validates configured models at startup

- Lists available models from each provider
- Tests model connectivity with minimal API calls
- Reports unavailable models with affected agents

## Voice Services

### VoiceService (`voice.py`)
**Purpose:** Multi-provider TTS with fallback chain

**Provider Priority:**
1. **ElevenLabs** (narration) - eleven_v3 multilingual, supports [audio tags]
2. **Speechify** (fallback) - simba-multilingual, Norwegian beta
3. **OpenAI** (dialogue) - gpt-4o-mini-tts with voice styling

| Method | Description |
|--------|-------------|
| `text_to_speech()` | Main TTS with use_case routing |
| `speech_to_text()` | Google Cloud or OpenAI Whisper |
| `text_to_speech_for_chapter()` | Convenience with SSML support |
| `generate_dialogue_with_audio()` | Single GPT-4o-audio call |

**Voice Settings:**

| Provider | Max Chars | Model | Features |
|----------|-----------|-------|----------|
| ElevenLabs | 4,800 | eleven_v3 | [audio tags], multilingual |
| Speechify | 1,900 | simba-multilingual | Norwegian beta (vegard) |
| OpenAI | 4,000 | tts-1 | Voice styling (alloy, nova, etc.) |

### SSMLProcessor (`ssml_processor.py`)
**Purpose:** SSML generation (deprecated - ElevenLabs handles natively)

- Dialogue detection with pitch/rate variations
- Dramatic pauses for punctuation
- Age-based pacing adjustments

## Classification Services

### InputClassifier (`input_classifier.py`)
**Purpose:** Classifies user input during story playback into 4 tiers

| Tier | Type | Target Chapter | Example |
|------|------|----------------|---------|
| **TIER_1** | Immediate | No impact | "What does navigator mean?" |
| **TIER_2** | Preference | Next chapter | "Make it scarier" |
| **TIER_3** | Story choice | Chapter N+2 | "Can she find a dragon?" |
| **TIER_4** | Addition | Chapter N+2+ | "Add pirates to the story" |

**Flow:** Pattern matching → LLM fallback (Foundry balanced mode)

### IntentClassifier (`intent_classifier.py`)
**Purpose:** Classifies initial conversation messages

| Intent | Action | Example |
|--------|--------|---------|
| `CONTINUE_STORY` | Resume active story | "What happens next?" |
| `NEW_STORY` | Create new story | "Tell me about dragons" |
| `EXPLORING` | Guide child | "I don't know" |
| `GREETING` | Respond warmly | "Hi!" |

## Utility Services

### EventEmitter (`events.py`)
**Purpose:** Real-time story progress events

| Event Type | When Emitted |
|------------|--------------|
| `story_created` | Story initialized |
| `structure_ready` | Outline complete |
| `character_ready` | Each character created |
| `chapter_ready` | Chapter approved |
| `story_interrupted` | User interrupted playback |
| `tts_completed` | Audio generation done |

### ValidationService (`validation_service.py`)
**Purpose:** JSON cleaning and repair for LLM output

- Removes markdown code blocks
- Fixes unescaped quotes (smart state machine)
- Closes truncated JSON
- Auto-fixes character data validation failures

### CharacterService (`character_service.py`)
**Purpose:** Norse-aware character identity matching

- Patronymic extraction (-sson, -dottir)
- Known spelling equivalents (Harald Fairhair = Harald Hårfagre)
- **CRITICAL:** Different patronymics = NEVER match

### ReviewService (`review_service.py`)
**Purpose:** Round Table feedback compilation

- Builds shared context for all reviewers
- Compiles revision guidance (MUST FIX, SHOULD IMPROVE, PRESERVE)
- Categorizes reviews by verdict

### TextProcessing (`text_processing.py`)
**Purpose:** Pure text utilities

- `normalize_norse_name()` - Unicode NFKD + accent removal
- `extract_patronymic()` - Extracts father's name
- `normalize_reviewer_name()` - Maps full names to short forms

### LoreLanternLogger (`logger.py`)
**Purpose:** Multi-mode logging system

| Mode | Output | Content |
|------|--------|---------|
| Terminal | stdout | Clean, emoji-decorated |
| Debug file | logs/*.log | Full detailed |
| JSON logs | logs/debug/*.json | Structured for analysis |

---

# 4. API Reference

## REST Endpoints

### Conversation Endpoints

#### `POST /api/conversation/start`
Initiate conversation with CompanionAgent.

**Request:**
```json
{
  "message": "I want a story about vikings",
  "child_id": "child_abc123",
  "language": "en"
}
```

**Response:**
```json
{
  "dialogue": "Oh, vikings! What an exciting choice...",
  "audio": "base64_audio_data",
  "intent": "new_story",
  "suggested_action": "show_story_setup",
  "child_name": "Emma",
  "child_age": 8
}
```

#### `POST /api/conversation/continue`
Continue exploring conversation.

**Request:** Same as `/start` + `conversation_turn` param

**Features:**
- Auto-selects story from library based on message matching
- Detects "new story" intent patterns
- **EARLY TRIGGERS:** Queues chapter/audio generation while greeting generates

#### `POST /api/conversation/init`
Initialize story generation.

**Request:**
```json
{
  "prompt": "A story about Harald Fairhair uniting Norway",
  "child_id": "child_abc123",
  "type": "historical",
  "language": "no",
  "target_age": 9,
  "preferences": {
    "educational_focus": "history",
    "difficulty": "medium",
    "themes": ["courage", "leadership"],
    "scary_level": "mild"
  },
  "chapters_to_write": 1,
  "pre_story_messages": ["I like battles", "Make it exciting"]
}
```

**Response:**
```json
{
  "success": true,
  "story_id": "story_xyz789",
  "welcome_message": "Let's create your viking adventure!",
  "story": { ... }
}
```

**Background:** Triggers `_generate_story_background()` which runs structure → characters → Chapter 1.

### Story Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stories/{id}` | GET | Get complete story |
| `/api/stories/{id}/chapters/{n}` | GET | Get specific chapter |
| `/api/stories/{id}/chapters/{n}/generate` | POST | Generate chapter on-demand |
| `/api/stories/{id}/generate-all` | POST | Generate ALL chapters (blocking) |
| `/api/stories/{id}/chapters/{n}/voice-direct` | POST | Generate SSML markup |
| `/api/stories/{id}/chapters/{n}/generate-audio` | POST | Generate/retrieve TTS audio |

#### `POST /api/stories/{id}/chapters/{n}/generate-audio`

**Response:**
```json
{
  "success": true,
  "audio_url": "https://storage.blob.../chapter_1.mp3?sv=...",
  "source": "blob_cache",
  "duration_estimate": 180,
  "provider": "ElevenLabs"
}
```

**Sources:** `blob_cache`, `blob_new`, `generated_base64`

### Profile Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/parents` | POST | Create parent account |
| `/api/parents` | GET | List all parents |
| `/api/parents/{id}` | GET | Get parent details |
| `/api/parents/{id}` | PATCH | Update parent |
| `/api/parents/{id}/children` | POST | Add child profile |
| `/api/parents/{id}/children` | GET | List children |
| `/api/children/{id}` | GET | Get child profile |
| `/api/children/{id}` | PATCH | Update child |
| `/api/children/{id}` | DELETE | Remove child (soft) |
| `/api/children/{id}/stories` | GET | Get story library |
| `/api/children/{id}/stories/{sid}/continue` | POST | Continue paused story |
| `/api/children/{id}/active-story` | GET | Get active story |

### Health Check

```
GET /api/health
→ {"status": "healthy", "service": "Lore Lantern", "coordinator_initialized": true}
```

## WebSocket

### Connection

```
ws://localhost:3000/ws/story/{story_id}
```

### Message Types (Client → Server)

| Type | Data | Handler |
|------|------|---------|
| `ping` | `{}` | Returns `pong` |
| `user_message` | `{"message": "..."}` | Classifies and routes input |
| `start_reading` | `{"chapter": 1}` | Updates reading state |
| `pause_reading` | `{"position": 0.5}` | Saves position |
| `resume_reading` | `{}` | Resumes playback |
| `finish_reading` | `{"chapter": 1}` | Triggers post-chapter discussion |
| `start_chapter` | `{"chapter_number": 1}` | Triggers prefetch |
| `audio_chunk` | `{"audio": "base64"}` | Speech-to-text processing |
| `interrupt` | `{"reason": "user_action"}` | Emits interrupt event |

### Message Types (Server → Client)

| Type | Description |
|------|-------------|
| `connection_established` | Connection confirmed |
| `dialogue_ready` | CompanionAgent response with audio |
| `chapter_started` | Chapter generation started |
| `transcript` | Speech-to-text result |
| `interrupt_acknowledged` | Interrupt confirmed |

---

# 5. Data Models

## Core Models

### Story
```python
Story(
    id: str,                          # "story_xyz789"
    prompt: str,                      # User's story request (max 20,000 chars)
    preferences: StoryPreferences,    # Language, difficulty, themes
    status: StoryStatus,              # INITIALIZING → COMPLETED
    structure: StoryStructure,        # Plot outline
    characters: List[Character],      # Full character profiles
    chapters: List[Chapter],          # Written content
    dialogues: List[DialogueEntry],   # Dialogue history
    metadata: StoryMetadata           # Timestamps, child_id
)
```

### StoryStructure
```python
StoryStructure(
    title: str,                       # "The Viking King"
    theme: str,                       # "courage and unity"
    chapters: List[ChapterOutline],   # Synopsis per chapter
    characters_needed: List[CharacterNeeded],
    educational_goals: List[EducationalGoal],
    estimated_reading_time_minutes: int,
    plot_elements: List[PlotElement], # Continuity tracking
    narrative_method: NarrativeMethod # POV strategy
)
```

### Chapter
```python
Chapter(
    number: int,
    title: str,
    synopsis: str,
    content: str,                     # Full prose (max 100,000 chars)
    characters_featured: List[str],
    educational_points: List[str],
    vocabulary_words: List[VocabularyWord],
    facts: List[VerifiedFact],
    word_count: int,
    reading_time_minutes: int,
    status: ChapterStatus,            # PENDING → READY → COMPLETED
    round_table_review: RoundTableReview,
    tts_content: str,                 # SSML-optimized narration
    audio_blob_url: str,              # Azure Blob SAS URL
    generation_metadata: GenerationMetadata
)
```

### Character
```python
Character(
    name: str,
    role: str,                        # protagonist, mentor, antagonist
    age: Union[int, str],             # 25 or "Ancient"
    background: str,                  # Min 50 words
    personality_traits: List[str],    # Min 2 traits
    motivation: str,                  # Psychological drive
    appearance: str,
    relationships: Dict[str, str],
    progression: CharacterProgression,
    character_arc: Dict[str, str]     # Chapter → milestone
)
```

### CharacterProgression
```python
CharacterProgression(
    skills_learned: List[CharacterSkill],
    personality_evolution: List[PersonalityEvolution],
    relationship_changes: List[RelationshipChange],
    current_emotional_state: str,     # Never "neutral"
    chapters_featured: List[int]
)
```

### CharacterSkill
```python
CharacterSkill(
    name: str,                        # "Leadership"
    level: int,                       # 1-10 (Novice → Master)
    acquired_chapter: int,            # 0 = backstory
    description: str
)
```

## Hybrid Generation Models

### QueuedInput
```python
QueuedInput(
    id: str,
    tier: InputTier,                  # TIER_1 through TIER_4
    raw_input: str,                   # Original user message
    classified_intent: str,           # LLM-extracted intent
    target_chapter: int,              # Which chapter this affects
    preference_updates: Dict,         # Tier 2 style changes
    story_direction: str,             # Tier 3 plot direction
    created_at: datetime,
    applied: bool,
    applied_at: datetime
)
```

### StoryReadingState
```python
StoryReadingState(
    story_id: str,
    session_id: str,
    current_chapter: int,
    chapter_position: float,          # 0.0-1.0 progress
    generating_chapter: int,
    chapter_statuses: Dict[str, ChapterStatus],
    queued_inputs: List[QueuedInput],
    playback_phase: PlaybackPhase,    # PRE/PLAYING/POST/TRANSITIONING
    discussion_started: bool
)
```

## Review Models

### AgentReview
```python
AgentReview(
    agent: str,                       # "Guillermo", "Bill", etc.
    domain: str,                      # "structure", "facts", etc.
    verdict: str,                     # "approve", "concern", "block"
    praise: str,
    concern: str,
    suggestion: str,
    chapter_ending_score: str,        # Stephen only: "hook", "adequate"
    tension_arc: str                  # Stephen only
)
```

### RoundTableReview
```python
RoundTableReview(
    decision: str,                    # "approved", "approved_with_notes", "revise"
    reviews: List[AgentReview],
    discussion: str,                  # Nnedi's response
    revision_guidance: str,           # Compiled for revision
    collective_notes: List[str],
    revision_rounds: int
)
```

## Profile Models

### ParentAccount
```python
ParentAccount(
    parent_id: str,
    language: str,                    # Family language
    child_ids: List[str],
    display_name: str,
    created_at: datetime
)
```

### ChildProfile
```python
ChildProfile(
    child_id: str,
    parent_id: str,
    name: str,                        # First name (1-50 chars)
    birth_year: int,
    story_ids: List[str],
    active_story_id: str,
    created_at: datetime
)
```

### LearningProgress (Backend-only)
```python
LearningProgress(
    child_id: str,
    vocabulary_bank: Dict[str, VocabularyEntry],
    concepts_mastered: Dict[str, ConceptMastery],
    reading_level: int,               # 1-10
    total_stories_completed: int,
    detected_interests: List[str],
    preferred_story_length: str,      # short, medium, long
    preferred_scary_level: str
)
```

---

# 6. Prompt System

## Prompt Files (17 total)

### Structure Prompts

| File | Purpose | Output |
|------|---------|--------|
| `generate_structure.py` | Create initial story outline | JSON with chapters, characters_needed, educational_goals |
| `refine_structure.py` | Update synopses after Chapter 1 | JSON with refined_chapters, skills_leveraged |

### Character Prompts

| File | Purpose | Output |
|------|---------|--------|
| `create_character.py` | Create detailed character profile | JSON with name, background, progression, arc |
| `validate_character.py` | Bill validates for duplicates | JSON with is_duplicate, is_historical, verdict |

### Writing Prompts

| File | Purpose | Output |
|------|---------|--------|
| `write_chapter.py` | Main chapter drafting (Nnedi) | JSON with content, vocabulary_words, facts |
| `revise_chapter.py` | Revise based on Round Table | Plain text narrative |
| `polish_chapter.py` | Final refinement polish | Plain text narrative |

### Review Prompts (Round Table)

| File | Agent | Evaluates |
|------|-------|-----------|
| `review_guillermo_structure.py` | Guillermo | Structure, pacing, thematic consistency |
| `review_bill_facts.py` | Bill | Historical accuracy, education, science |
| `review_clarissa_characters.py` | Clarissa | Voice consistency, psychology, arc |
| `review_benjamin_prose.py` | Benjamin | Rhythm, show-don't-tell, sensory grounding |
| `review_stephen_tension.py` | Stephen | Hooks, tension architecture, momentum |
| `review_continuity_threads.py` | Continuity | Plot elements, resolutions, danglers |

### Narrator Prompts

| File | Purpose | Output |
|------|---------|--------|
| `narrative_method_debate.py` | Guillermo + Stephen decide POV | JSON with method, pov_characters, hook_strategy |
| `discussion.py` | Nnedi responds to concerns | JSON with responses, revision_plan |
| `commentary.py` | Real-time milestone commentary | Plain text in target language |

## Narrative Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `LINEAR_SINGLE_POV` | One protagonist throughout (Harry Potter) | Ages 5-10, simple adventures |
| `LINEAR_DUAL_THREAD` | Two storylines merge (Percy Jackson) | Ages 8-12, quests |
| `MULTI_POV_ALTERNATING` | Multiple perspectives (Da Vinci Code) | Ages 10+, complex plots |
| `FRAME_NARRATIVE` | Story within story (Princess Bride) | Ages 8+, fairy tales |

## Sensory Grounding Requirement

Every paragraph MUST include ≥1 sensory detail:

| Score | Definition | Action |
|-------|------------|--------|
| `strong` | Every paragraph grounded | Approve |
| `adequate` | 1-2 paragraphs need work | Concern |
| `needs_work` | 3+ paragraphs missing | BLOCK |

## Review Output Format

All reviewers output:
```json
{
  "agent": "AgentName",
  "domain": "domain_name",
  "verdict": "approve|concern|block",
  "praise": "What works...",
  "concern": "What troubles...",
  "suggestion": "How to improve..."
}
```

---

# 7. Configuration

## Environment Variables

### LLM API Keys
```bash
CLAUDE_API_KEY=sk-ant-...           # Anthropic (optional)
OPENAI_API_KEY=sk-proj-...          # OpenAI (required for TTS)
GOOGLE_GENAI_API_KEY=AIza...        # Google Gemini (required)
```

### Azure Infrastructure
```bash
# Azure SQL Database
AZURE_SQL_SERVER=lorelantern-db.database.windows.net
AZURE_SQL_DATABASE=lorelantern
AZURE_SQL_USERNAME=lorelantern_admin
AZURE_SQL_PASSWORD=***

# Azure Blob Storage
AZURE_BLOB_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
AZURE_BLOB_CONTAINER=lorelantern-audio

# Feature Flag
USE_AZURE_SQL=true                  # Enable Azure SQL (default: false/Firebase)
```

### Azure AI Foundry (Model Router)
```bash
FOUNDRY_ENDPOINT=https://foundry-lorelantern.cognitiveservices.azure.com
FOUNDRY_API_KEY=***
FOUNDRY_API_VERSION=2024-12-01-preview
USE_FOUNDRY=true                    # Enable Foundry routing
FOUNDRY_ROUTING_MODE=quality        # quality|cost|balanced
```

### Azure Claude Deployment
```bash
AZURE_CLAUDE_ENDPOINT=https://azure-claude.cognitiveservices.azure.com
AZURE_CLAUDE_API_KEY=***
AZURE_CLAUDE_MODEL=claude-sonnet-4-5
```

### TTS Providers
```bash
ELEVENLABS_API_KEY=sk_...           # ElevenLabs (premium narration)
SPEECHIFY_API_KEY=xCW...            # Speechify (fallback)
USE_ELEVENLABS=true                 # Enable ElevenLabs
DISABLE_TTS=false                   # Disable all TTS (dev)
```

### Firebase (Legacy - Deprecated)
```bash
FIREBASE_API_KEY=***
FIREBASE_AUTH_DOMAIN=***.firebaseapp.com
FIREBASE_DATABASE_URL=https://***.firebasedatabase.app
FIREBASE_PROJECT_ID=***
FIREBASE_CLIENT_EMAIL=***@***.iam.gserviceaccount.com
FIREBASE_PRIVATE_KEY=-----BEGIN PRIVATE KEY-----...
```

### Application
```bash
PORT=3000
LOG_LEVEL=INFO                      # INFO|DEBUG|WARNING|ERROR
CORS_ALLOWED_ORIGINS=*              # Or comma-separated: https://a.com,https://b.com
DEV_MODE=false
```

### A/B Testing
```bash
TEST_NARRATIVE_MODEL=gpt-4o         # Override narrative agent model
TEST_STRUCTURE_MODEL=gemini-2.0     # Override structure agent model
```

### Debug Logging
```bash
DEBUG_AGENT_IO=true                 # Log agent inputs/outputs
DEBUG_STORAGE=true                  # Log storage operations
DEBUG_API_CALLS=true                # Log LLM calls
DEBUG_LOG_DIR=logs/debug
DEBUG_LOG_FORMAT=text               # text|json
```

## Validation Limits

```python
# User Input (src/config/limits.py)
PROMPT_MAX_LENGTH = 20_000          # Story prompts + reference material
MESSAGE_MAX_LENGTH = 5_000          # Chat messages

# Generated Content
CHAPTER_CONTENT_MAX_LENGTH = 100_000
SYNOPSIS_MAX_LENGTH = 10_000
TITLE_MAX_LENGTH = 100
TITLE_MIN_LENGTH = 3
NAME_MAX_LENGTH = 100
NAME_MIN_LENGTH = 2

# Collections
MAX_CHAPTERS_DEFAULT = 20
MAX_CHARACTERS_DEFAULT = 50
MAX_DIALOGUE_HISTORY = 100
```

## Language Support

| Language | Code | TTS Code | Voice (ElevenLabs) |
|----------|------|----------|-------------------|
| English | `en` | `en-US` | `7p1Ofvcwsv7UBPoFNcpI` |
| Norwegian Bokmål | `no` | `nb-NO` | `xF681s0UeE04gsf0mVsJ` |
| Spanish | `es` | `es-ES` | `21m00Tcm4TlvDq8ikWAM` |

---

## File Index

| Directory | Files | Purpose |
|-----------|-------|---------|
| `src/agents/` | 11 | Agent definitions |
| `src/api/` | 2 | REST + WebSocket handlers |
| `src/config/` | 3 | Settings, limits |
| `src/crew/` | 1 | Coordinator orchestration |
| `src/models/` | 2 | Pydantic data models |
| `src/prompts/` | 17 | LLM prompt templates |
| `src/services/` | 17 | Business logic services |
| `src/utils/` | 1 | Language utilities |
| `src/` | 1 | main.py entry point |
