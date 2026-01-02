# Lore Lantern - Architecture Rules & Standards

**Last Updated:** December 2025
**Version:** 2.0

## 📋 Purpose
This document defines the architectural standards and patterns that MUST be followed across the entire codebase to ensure consistency, reliability, and maintainability.

---

## 🌐 Technology Stack Overview

| Layer | Technology | Purpose |
|-------|------------|---------|
| **LLM Providers** | Azure AI Foundry Model Router, Claude Sonnet 4.5 | Story generation, agent intelligence |
| **Database** | Azure SQL Database | Structured data (stories, characters, chapters) |
| **Audio Storage** | Azure Blob Storage | MP3 chapter audio files |
| **TTS Narration** | ElevenLabs (eleven_v3), Speechify | Story chapter audio |
| **TTS Dialogue** | OpenAI gpt-4o-mini-tts | CompanionAgent voice |
| **Direct Audio** | OpenAI gpt-4o-mini-audio-preview | Single-call LLM+TTS |
| **Backend** | FastAPI + Python | API and WebSocket server |
| **Agent Framework** | CrewAI + LiteLLM | Multi-agent orchestration |

---

## 🔧 1. Logger API Standards

### 1.1 StorytellerLogger Methods

The custom logger (`src/services/logger.py`) has these methods:

```python
# ✅ CORRECT USAGE
logger.info(message: str)
logger.debug(component: str, message: str)
logger.error(component: str, message: str, error: Exception = None)
logger.warning(message: str)
logger.job_received(job_type: str, story_id: str, details: str)
logger.job_completed(job_type: str, story_id: str, duration: float)
logger.job_failed(job_type: str, story_id: str, error: str)
logger.agent_working(agent_name: str, task: str)
logger.agent_completed(agent_name: str, task: str, duration: float)
logger.dialogue_output(story_id: str, speaker: str, message: str, phase: str = "")
```

### 1.2 Common Mistakes to AVOID

```python
# ❌ WRONG - logger.exception() does NOT exist
self.logger.exception(f"Error: {e}")

# ❌ WRONG - debug() requires component parameter
self.logger.debug(f"Some debug message")

# ❌ WRONG - error() requires component parameter
self.logger.error(f"❌ Failed: {e}")

# ✅ CORRECT
self.logger.error("ComponentName", f"Failed to process", e)
self.logger.debug("ComponentName", "Debug information")
```

---

## 🏗️ 2. Agent Communication Pattern

All agent interactions MUST follow this standard flow:

### 2.1 Agent Execution Pattern

```python
# 1. Create Task with detailed description
task = Task(
    description="...",  # Clear instructions
    agent=self.agent_name,
    expected_output="Valid JSON with specific format"
)

# 2. Execute with CrewAI
crew = Crew(agents=[self.agent_name], tasks=[task], process=Process.sequential, verbose=True)
result = crew.kickoff()

# 3. Log raw output
self.logger.info(f"Agent completed")
self.logger.debug("AgentName", f"Raw output: {len(str(result))} chars")

# 4. Clean markdown code blocks
result_str = self._clean_json_output(str(result))

# 5. Parse JSON
try:
    data = json.loads(result_str)
    self.logger.info(f"✅ Successfully parsed JSON")
except json.JSONDecodeError as e:
    self.logger.error("AgentName", f"JSON parsing failed", e)
    # Save debug file
    debug_file = Path("logs/debug") / f"failed_{type}_story_{id}.txt"
    debug_file.write_text(str(result))
    self.logger.error("AgentName", f"Raw output saved to: {debug_file}")
    return {"success": False, "error": str(e)}

# 6. Validate with Pydantic
model_obj = PydanticModel(**data)

# 7. Save to Firebase
await self.firebase.save_xxx(story_id, model_obj)
self.logger.info(f"💾 Saved to Firebase")

# 8. Emit WebSocket event
await story_events.emit("event_type", story_id, event_data)
```

---

## 📊 3. Data Type Standards

### 3.1 Pydantic Model Field Types

These are the **required** data types per `ASSETS/schemas/models.py`:

```python
# Character model
character_arc: Optional[Dict[str, str]]  # NOT List!
personality_traits: List[str]
relationships: Dict[str, str]

# Story structure
arc_milestones: Optional[Dict[str, str]]  # NOT List!
chapters: List[ChapterOutline]
characters_needed: List[CharacterNeeded]
```

### 3.2 Common Conversion Fixes

```python
# FIX: Convert character_arc from list to dict (GPT-5 sometimes generates wrong format)
if 'character_arc' in data and isinstance(data['character_arc'], list):
    arc_list = data['character_arc']
    if arc_list and all(arc_list):
        data['character_arc'] = {str(i+1): arc_list[i] for i in range(len(arc_list)) if arc_list[i]}
    else:
        data['character_arc'] = None
```

---

## 🚨 4. Error Handling Pattern

### 4.1 Standard Error Handling Template

```python
try:
    # Agent execution
    result = await some_agent_operation()

    if not result["success"]:
        error_msg = result.get('error', 'Unknown error')
        self.logger.error("Component", error_msg)

        # Update Firebase status
        await self.firebase.update_story_status(story_id, StoryStatus.FAILED.value)

        # Emit WebSocket error event
        await story_events.emit("error", story_id, {
            "stage": "operation_name",
            "error": error_msg,
            "message": "User-friendly error message"
        })
        return

except json.JSONDecodeError as e:
    self.logger.error("Component", f"JSON parsing failed", e)
    self.logger.error("Component", f"Error position: line {e.lineno}, column {e.colno}")

    # Save debug file
    debug_file = Path("logs/debug") / f"failed_{operation}_{story_id}.txt"
    debug_file.parent.mkdir(parents=True, exist_ok=True)
    debug_file.write_text(raw_output, encoding='utf-8')
    self.logger.error("Component", f"Raw output saved to: {debug_file}")

except Exception as e:
    self.logger.error("Component", f"Unexpected error in operation", e)
    await story_events.emit("error", story_id, {
        "stage": "operation_name",
        "error": str(e),
        "error_type": type(e).__name__,
        "message": "An unexpected error occurred"
    })
```

### 4.2 WebSocket Error Event Structure

```python
{
    "stage": "structure_generation | character_creation | chapter_writing",
    "error": "Technical error message",
    "message": "User-friendly explanation",
    "chapter_number": 1  # Optional, if applicable
}
```

---

## 🔄 5. Data Flow Architecture

### 5.1 Story Creation Flow

```
Frontend (Browser)
    ↓ HTTP POST /api/conversation/init
API Routes
    ↓ coordinator.initialize_story()
CompanionAgent → Immediate response (via Direct Audio or OpenAI TTS)
    ↓ Background task starts
    ↓
StructureAgent → Generate story outline
    ↓ Save to Azure SQL
    ↓ Emit WebSocket: structure_ready
    ↓
CharacterAgent → For each character
    ↓ Save to Azure SQL
    ↓ Emit WebSocket: character_ready
    ↓
NarrativeAgent → For each chapter
    ↓ Writers' Round Table → 4 parallel reviewers
    ↓ (Revise if blocked, max 3 rounds)
    ↓ VoiceDirectorAgent → Add audio tags for expressiveness
    ↓ TTS Generation → ElevenLabs/Speechify
    ↓ Audio → Azure Blob Storage
    ↓ Save to Azure SQL
    ↓ Emit WebSocket: chapter_ready
```

### 5.2 Separation of Concerns

| Layer | Responsibility | Files |
|-------|---------------|-------|
| **API** | HTTP endpoints, background tasks | `src/api/routes.py`, `src/api/websocket.py` |
| **Orchestration** | Agent coordination, workflow | `src/crew/coordinator.py` |
| **Agents** | AI task execution | `src/agents/*.py` |
| **Services** | Database, voice, logging, events | `src/services/*.py` |
| **Models** | Data validation, types | `src/models/*.py` |

### 5.3 Database Architecture

**Azure SQL Database** stores all structured data:
- Stories, chapters, characters
- User profiles, preferences
- Reading state, progress tracking

**Azure Blob Storage** stores audio files:
- Chapter narration MP3s
- Container: `lorelantern-audio`

---

## 🎯 6. Agent-Specific Standards

### LLM Model Configuration

All agents use **Claude Sonnet 4.5** (`claude-sonnet-4-5-20250929`) via LiteLLM, with Azure Foundry Model Router as an alternative routing option.

```python
# From src/config/settings.py
agent_models = {
    "dialogue": "claude-sonnet-4-5-20250929",
    "structure": "claude-sonnet-4-5-20250929",
    "character": "claude-sonnet-4-5-20250929",
    "narrative": "claude-sonnet-4-5-20250929",
    "factcheck": "claude-sonnet-4-5-20250929",
    "voice_director": "claude-sonnet-4-5-20250929"
}
```

### 6.1 CompanionAgent (Front-Face)
- **Model**: Gemini Flash (fast) + Claude Sonnet (spotlights)
- **Output**: Plain text dialogue + audio
- **Purpose**: Always-available user companion, never blocked by CrewAI
- **TTS**: OpenAI gpt-4o-mini-tts or Direct Audio (gpt-4o-mini-audio-preview)

### 6.2 StructureAgent
- **Model**: Claude Sonnet 4.5 (or Azure Model Router)
- **Output**: JSON with chapters, characters_needed, educational_goals
- **Critical**: Must complete full JSON (avoid truncation)
- **Validation**: Save debug file if JSON fails

### 6.3 CharacterAgent
- **Model**: Claude Sonnet 4.5
- **Output**: JSON with character profile
- **Data Fix**: Convert character_arc list→dict if needed
- **Validation**: Ensure all required Pydantic fields present

### 6.4 NarrativeAgent
- **Model**: Claude Sonnet 4.5
- **Output**: JSON with chapter content
- **Context**: Receives full character profiles for rich prose
- **Validation**: Check word count, vocabulary words

### 6.5 Writers' Round Table (4 Reviewers)
- **Models**: Claude Sonnet 4.5 (all reviewers)
- **Execution**: Parallel via asyncio.gather()
- **Reviewers**: Guillermo (structure), Bill (facts), Clarissa (characters), Benjamin (prose)
- **Verdicts**: approve | concern | block

### 6.6 VoiceDirectorAgent
- **Model**: Claude Sonnet 4.5
- **Output**: Text with ElevenLabs audio tags
- **Purpose**: Add expressiveness for narration ([whispers], [laughing], etc.)
- **TTS Target**: ElevenLabs eleven_v3 model

---

## ✅ 7. Validation Checklist

Before committing code, verify:

- [ ] All `logger.debug()` calls have component parameter
- [ ] All `logger.error()` calls have component parameter
- [ ] No `logger.exception()` calls (doesn't exist)
- [ ] All agent executions have try/except blocks
- [ ] All JSON errors save debug files
- [ ] All operations emit WebSocket events
- [ ] All data types match Pydantic schemas
- [ ] All character_arc fields are Dict not List
- [ ] All Firebase operations have error handling
- [ ] All user-facing errors have friendly messages

---

## 🔍 8. Debugging Standards

### 8.1 Debug File Naming
```
logs/debug/failed_structure_story_{story_id}.txt
logs/debug/failed_character_{char_name}_story_{story_id}.txt
logs/debug/failed_chapter_{num}_story_{story_id}.txt
```

### 8.2 Debug Logging Requirements

When `DEBUG_MODE=true`:
- Log raw agent outputs (first/last 500 chars)
- Log Firebase read/write operations
- Log task descriptions sent to agents
- Log data transformations (list→dict conversions)
- Log validation failures with context

---

## 📚 9. Reference Implementation

Use `src/crew/coordinator.py` methods as reference:
- ✅ `generate_story_structure()` - Proper error handling
- ✅ `create_characters()` - Data type conversion
- ✅ `write_chapter()` - Full agent execution pattern

---

## 🚀 10. Performance Standards

- Structure generation: < 30 seconds
- Character creation: < 10 seconds per character
- Chapter writing: < 60 seconds per chapter
- Total story (3 chapters): < 5 minutes

---

**Last Updated**: 2025-12-28
**Maintainer**: Development Team
**Version**: 2.0
