# Architecture Decisions

This document explains the key architectural decisions in Lore Lantern and the rationale behind them. These are intentional trade-offs, not technical debt.

---

## 1. Business Logic in API Routes (`routes.py`)

**Decision:** Story orchestration logic lives in `api/routes.py` rather than a separate service layer.

**Rationale:**
- **Performance**: Avoiding extra async function layers reduces latency for real-time story generation
- **Simplicity**: Direct call chains are easier to trace and debug when issues arise
- **WebSocket Integration**: Events need to emit from the API layer where WebSocket connections are managed
- **MVP Focus**: For a team of 1-2 developers, an extra abstraction layer adds cognitive overhead without proportional benefit

**When to revisit:**
- When the team grows beyond 3 developers
- When unit test coverage becomes a priority (extract for testability)
- When multiple entry points (CLI, scheduled jobs) need the same orchestration logic

---

## 2. Large Coordinator File (`coordinator.py` - 4,500 lines)

**Decision:** Keep all agent orchestration in a single `StoryCrewCoordinator` class.

**Rationale:**
- **Single Orchestration Point**: Coordinates 15+ AI agents with complex inter-dependencies
- **Prompt Extraction Done**: Reduced from 6,400 lines by extracting prompts to `/src/prompts/`
- **Cohesion**: All story generation phases (structure, characters, chapters, review) share state and context
- **Debugging**: Single file makes it easier to trace the full story generation flow

**What was already done:**
- Extracted ~1,800 lines of prompts to dedicated prompt files
- Organized prompts into `/src/prompts/{agent_type}/` directories
- Each prompt is now a function that accepts context and returns formatted strings

**When to revisit:**
- When adding new story types (e.g., non-fiction, poetry) that need different flows
- When the file exceeds 6,000 lines again
- Consider sub-coordinators: `StructureCoordinator`, `CharacterCoordinator`, `ChapterCoordinator`

---

## 3. VoiceService Handles Multiple TTS Providers (1,400 lines)

**Decision:** Single `VoiceService` class manages 5 TTS providers (OpenAI, ElevenLabs, Speechify, Google, Azure).

**Rationale:**
- **Fallback Logic Centralized**: If ElevenLabs fails, fallback to Speechify, then OpenAI
- **Provider Selection is Complex**: Depends on language, voice quality requirements, cost
- **Unified Interface**: All callers use `voice_service.text_to_speech()` regardless of provider
- **Rate Limiting**: Centralized retry/backoff logic across all providers

**When to revisit:**
- When adding more than 2 new TTS providers
- When provider-specific features need isolation (e.g., ElevenLabs voice cloning)
- Consider: `TTSProviderFactory` with provider-specific classes

---

## 4. DatabaseService Handles Multiple Entity Types (1,500 lines)

**Decision:** Single `DatabaseService` manages stories, chapters, characters, dialogues, learning progress, and user profiles.

**Rationale:**
- **Single Connection Pool**: Efficient database connection management
- **Transaction Boundaries**: Some operations need cross-entity transactions
- **Azure SQL Migration**: Currently migrating from Firebase to Azure SQL; single service simplifies this
- **Query Optimization**: Complex queries that join multiple entities are easier to optimize in one place

**When to revisit:**
- After Azure SQL migration is complete and stable
- When the team grows and different developers own different data domains
- Consider: Repository pattern with `StoryRepository`, `ChapterRepository`, etc.

---

## 5. CORS Allows All Origins (`*`)

**Decision:** Default CORS policy allows all origins.

**Rationale:**
- **Public API**: Lore Lantern is designed to be accessed by the physical lantern device
- **Multiple Clients**: Web app, mobile apps, embedded device - all need API access
- **Authentication via Tokens**: Security is handled by Firebase authentication tokens, not origin restrictions
- **Configurable**: `CORS_ALLOWED_ORIGINS` env var allows restriction when needed

**When to restrict:**
- If building a web-only product with known frontend domains
- If API abuse becomes a concern (combine with rate limiting)

---

## 6. Global Service Singletons in `main.py`

**Decision:** Services are instantiated as module-level globals in `main.py`.

**Rationale:**
- **Startup Initialization**: Services are initialized once during app startup via `lifespan`
- **Shared State**: WebSocket handlers and HTTP routes need access to the same service instances
- **FastAPI Pattern**: Common pattern in FastAPI applications for service management

**When to revisit:**
- When parallel test execution is needed (tests can't share global state)
- When implementing dependency injection for better testability
- Consider: `python-dependency-injector` or custom `DIContainer`

---

## 7. Duplicate-Looking Classification Systems

**Observation:** Two classification systems exist: `IntentClassifier` and `InputClassifier`.

**Rationale:**
- **Different Purposes**: Intent classification (what user wants) vs. input tiering (how complex is the input)
- **Different Callers**: Intent used for routing, tiering used for LLM model selection
- **Intentional Separation**: Kept separate to allow independent evolution

**When to merge:**
- If both always run together and results are always combined
- If maintenance becomes a burden

---

## Future Improvements (When Scaling)

These improvements are documented for when the project scales:

| Improvement | Trigger | Benefit |
|-------------|---------|---------|
| Extract `StoryOrchestrationService` | Adding test coverage | Testability |
| TTS Provider pattern | Adding 2+ new providers | Maintainability |
| Repository pattern | Team grows to 3+ | Clean separation |
| Dependency injection | Parallel test runs needed | Test isolation |
| Sub-coordinators | Adding new story types | Modularity |

---

## Summary

The current architecture prioritizes:
1. **Simplicity** over theoretical purity
2. **Performance** over abstraction layers
3. **Debuggability** over separation
4. **MVP velocity** over enterprise patterns

These are intentional choices for a small team building a novel product. The architecture can evolve as the team and product grow.
