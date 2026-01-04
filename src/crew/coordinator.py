"""
Crew Coordinator - Orchestrates all agents using CrewAI

This is the "round table" where all agents work together to create stories.
"""

from crewai import Crew, Task, Process
from pydantic import ValidationError
from typing import Dict, Any, List, Optional, Tuple
import json
import re
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
import unicodedata
from pathlib import Path

# Dedicated thread pool for CrewAI operations
# Isolated from default pool to prevent blocking dialogue and other async operations
CREWAI_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="crewai")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents import (
    create_dialogue_agent,
    create_structure_agent,
    create_character_agent,
    create_narrative_agent,
    create_factcheck_agent,
    create_voice_director_agent,
    create_tension_agent
)
from src.agents.line_editor import create_line_editor_agent
from src.agents.continuity import create_continuity_agent
from src.services import LLMService
# Storage service can be DatabaseService (Azure SQL) or FirebaseService (legacy)
from src.services.events import (
    story_events,
    EVENT_AGENT_STARTED,
    EVENT_AGENT_COMPLETED,
    EVENT_REVIEWER_VERDICT
)
from src.services.model_tracker import ModelTracker
from src.services.adaptive_rate_limiter import get_rate_limiter, RateLimitError
from src.services.logger import get_logger
from src.models import (
    Story, StoryStructure, Character, Chapter, ChapterOutline,
    StoryStatus, DialogueEntry, ChapterStatus, QueuedInput,
    AgentReview, RoundTableReview,
    VoiceDirectionMetadata, CharacterVoiceMapping, EmotionalBeat,
    GenerationMetadata
)
from datetime import datetime
from src.utils.language_styles import get_language_style, get_prose_instruction, get_tts_language_code
from src.services.text_processing import (
    normalize_reviewer_name,
    normalize_norse_name,
    extract_patronymic,
    extract_first_name
)
from src.services.validation_service import (
    clean_json_output,
    clean_json_for_character,
    auto_fix_character_data
)
from src.services.character_service import (
    normalize_character_names,
    check_relationship_match,
    match_character_identity,
    CharacterService
)
from src.services.review_service import (
    build_review_context,
    compile_revision_guidance
)
from src.prompts.reviews import (
    get_guillermo_structure_prompt,
    get_bill_facts_prompt,
    get_clarissa_characters_prompt,
    get_benjamin_prose_prompt,
    get_stephen_tension_prompt,
    get_continuity_threads_prompt
)
from src.prompts.writing import (
    get_write_chapter_prompt,
    get_write_chapter_json_schema,
    get_revise_chapter_prompt,
    get_polish_chapter_prompt
)
from src.prompts.structure import (
    get_generate_structure_prompt,
    get_refine_structure_prompt,
    get_refine_structure_system_prompt
)
from src.prompts.character import (
    get_create_character_prompt,
    get_validate_character_prompt
)
from src.prompts.narrator import (
    get_narrative_method_debate_prompt,
    get_discussion_prompt,
    get_structure_ready_commentary_prompt,
    get_character_ready_commentary_prompt,
    get_character_ready_fallback_prompt,
    get_chapter_ready_commentary_prompt,
    get_chapter_ready_fallback_prompt,
    get_fallback_commentary_prompt
)
import time

class StoryCrewCoordinator:
    """
    Coordinates all agents using CrewAI to create stories.

    This is the main orchestrator that handles the complete story creation workflow.
    """

    def __init__(self, storage_service, logger=None):
        """
        Initialize the coordinator.

        Args:
            storage_service: Storage service instance (DatabaseService for Azure SQL, or FirebaseService)
            logger: Optional logger instance for debug logging

        Agent model configuration is now centralized in src/config/models.yaml
        and managed by the LLM Router. Agents self-configure from the router.
        """
        from src.services.logger import get_logger
        from src.services.llm_router import get_llm_router

        self.storage = storage_service
        self.logger = logger if logger else get_logger()

        # Get the LLM Router for logging active configuration
        self.router = get_llm_router()

        # Log active overrides (group and agent level)
        for group_name in ["roundtable", "writers", "post"]:
            group_cfg = self.router.config.get("groups", {}).get(group_name, {})
            if group_cfg.get("model"):
                self.logger.info(f"🔬 Group override: {group_name} → {group_cfg['model']}")

        # Rate limiting for Foundry Model Router
        # Limit concurrent API calls to prevent 429 Too Many Requests
        # Model Router has TPM (tokens per minute) limits that can be exceeded
        # when running multiple agents in parallel
        self._foundry_semaphore = asyncio.Semaphore(2)  # Max 2 concurrent Foundry calls
        self._foundry_delay = 2.0  # Seconds between releasing semaphore (increased from 1.0 to reduce 429s)

        # Create agents - they self-configure from LLM Router (models.yaml)
        # Small delays between agent creations prevent CrewAI I18N race condition
        # when multiple agents try to load translations/en.json concurrently

        self.dialogue_agent = create_dialogue_agent()
        time.sleep(0.05)
        self.structure_agent = create_structure_agent()
        time.sleep(0.05)
        self.character_agent = create_character_agent()
        time.sleep(0.05)
        self.narrative_agent = create_narrative_agent()
        time.sleep(0.05)
        self.factcheck_agent = create_factcheck_agent()
        time.sleep(0.05)
        self.line_editor_agent = create_line_editor_agent()
        time.sleep(0.05)
        self.voice_director_agent = create_voice_director_agent()
        time.sleep(0.05)
        self.continuity_agent = create_continuity_agent()
        time.sleep(0.05)
        self.tension_agent = create_tension_agent()

        self.all_agents = [
            self.dialogue_agent,
            self.structure_agent,
            self.character_agent,
            self.narrative_agent,
            self.factcheck_agent,
            self.line_editor_agent,
            self.voice_director_agent,
            self.continuity_agent,
            self.tension_agent
        ]

        # Initialize CharacterService with dependencies
        # The LLM caller wraps Foundry service for character deduplication
        self.character_service = CharacterService(
            storage=storage_service,
            llm_caller=self._create_llm_caller(),
            app_logger=self.logger
        )

    def _create_llm_caller(self):
        """
        Create an LLM caller function for CharacterService.

        Returns an async callable that wraps Azure AI Foundry chat completion
        with rate limiting and proper error handling.

        Returns:
            Async callable: (prompt, system_prompt, model_preference) -> response_text
        """
        async def llm_caller(prompt: str, system_prompt: str, model_preference: str) -> str:
            from src.services.foundry import get_foundry_service

            foundry = get_foundry_service()
            if not foundry:
                raise RuntimeError("Foundry service not available")

            # Map preference to routing mode
            routing_mode = "balanced" if model_preference == "balanced" else "quality"

            async with self._foundry_semaphore:
                response = await asyncio.wait_for(
                    foundry.chat_completion(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        routing_mode=routing_mode,
                        max_tokens=2000,
                        temperature=0.3  # Lower temperature for factual dedup task
                    ),
                    timeout=60.0
                )
                await asyncio.sleep(self._foundry_delay)

            return response.get("content", "").strip()

        return llm_caller

    async def _run_crew_async(self, crew: Crew, timeout_seconds: int = 300):
        """
        Run CrewAI kickoff in a dedicated thread pool to avoid blocking the event loop.

        CrewAI's crew.kickoff() is synchronous and can take minutes for complex tasks.
        This wrapper uses CREWAI_EXECUTOR (dedicated 8-thread pool) to isolate heavy
        agent work from dialogue and other async operations.

        Args:
            crew: CrewAI Crew instance to execute
            timeout_seconds: Maximum time to wait for completion (default 5 minutes)

        Returns:
            The result from crew.kickoff()

        Raises:
            asyncio.TimeoutError: If the operation takes longer than timeout_seconds
        """
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(CREWAI_EXECUTOR, crew.kickoff),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            self.logger.error(f"CrewAI operation timed out after {timeout_seconds}s")
            raise

    async def _plan_narrative_method(
        self,
        story: "Story",
        structure_data: Dict[str, Any]
    ) -> "NarrativeMethod":
        """
        Guillermo + Stephen debate: Determine the narrative storytelling method.

        This collaborative planning phase happens AFTER structure is created but
        BEFORE chapter writing begins. The two agents discuss which narrative
        method best suits the story.

        Methods:
        - linear_single_pov: Single protagonist, linear time (Harry Potter style)
        - linear_dual_thread: Two storylines that merge
        - multi_pov_alternating: Multiple POVs per chapter (Da Vinci Code style)
        - frame_narrative: Story within story

        Args:
            story: Story object with preferences
            structure_data: The story structure dict from StructureAgent

        Returns:
            NarrativeMethod with chosen method, POV characters, and hook strategy
        """
        from src.models.models import NarrativeMethod

        target_age = story.preferences.target_age if story.preferences.target_age else 10
        theme = structure_data.get("theme", "adventure")
        chapters = structure_data.get("chapters", [])
        characters = structure_data.get("characters_needed", [])
        title = structure_data.get("title", "Untitled")

        # Find protagonist(s)
        protagonists = [c["name"] for c in characters if c.get("role") == "protagonist"]
        if not protagonists:
            protagonists = [characters[0]["name"]] if characters else ["Main Character"]

        # Build character list for debate context
        char_summary = "\n".join([
            f"- {c['name']} ({c.get('role', 'unknown')}): {c.get('importance', 'supporting')}"
            for c in characters
        ])

        # Generate debate prompt using extracted prompt function
        debate_prompt = get_narrative_method_debate_prompt(
            title=title,
            theme=theme,
            target_age=target_age,
            chapters=chapters,
            char_summary=char_summary
        )

        # Use structure agent for this planning task (Guillermo leads, Stephen advises)
        task = Task(
            description=debate_prompt,
            expected_output="JSON narrative method selection",
            agent=self.structure_agent
        )

        crew = Crew(
            agents=[self.structure_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )

        try:
            result = await self._run_crew_async(crew, timeout_seconds=120)
            result_str = clean_json_output(str(result).strip())
            method_data = json.loads(result_str)

            self.logger.info(f"📖 Narrative method selected: {method_data.get('method', 'linear_single_pov')}")
            self.logger.info(f"   POV: {method_data.get('pov_characters', protagonists)}")
            self.logger.info(f"   Hook strategy: {method_data.get('hook_strategy', 'chapter-ending hooks')[:50]}...")

            return NarrativeMethod(
                method=method_data.get("method", "linear_single_pov"),
                pov_characters=method_data.get("pov_characters", protagonists),
                hook_strategy=method_data.get("hook_strategy", "Each chapter ends with unanswered question or discovery"),
                chapter_rhythm=method_data.get("chapter_rhythm", "alternating action/quiet"),
                rationale=method_data.get("rationale", "Selected for story type and target audience")
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Narrative method planning failed: {e}. Using default.")
            # Default to simple linear POV for safety
            return NarrativeMethod(
                method="linear_single_pov",
                pov_characters=protagonists,
                hook_strategy="Each chapter ends with unanswered question or discovery",
                chapter_rhythm="alternating action/quiet",
                rationale=f"Default for age {target_age} audience"
            )

    async def initialize_story(
        self,
        prompt: str,
        user_id: str = None,
        target_age: int = None,
        preferences: Dict[str, Any] = None,
        child_id: str = None,
        language: str = None
    ) -> Dict[str, Any]:
        """
        Initialize a new story with user prompt.

        This handles the initial dialogue response PLUS kicks off background story generation.

        Args:
            prompt: User's story request
            user_id: DEPRECATED - Use child_id. User ID for backward compatibility.
            target_age: Target age for content. If not provided, derived from child profile.
            preferences: Story preferences
            child_id: Child profile ID (new family account model)
            language: Language override. If not provided, fetched from parent account.

        Returns:
            Dict with story_id, welcome_message, and initial story data
        """
        start_time = time.time()

        # Handle new child_id model vs legacy user_id
        effective_child_id = child_id or user_id
        effective_target_age = target_age or 10  # Default
        effective_language = language or "en"    # Default

        # If child_id provided, fetch child profile and parent account
        # This GROUNDS the story to the child's actual age from their profile
        if child_id:
            try:
                child = await self.storage.get_child_profile(child_id)
                if child:
                    # GROUND target_age from child's birth_year (calculated as current_age)
                    if target_age is None:
                        effective_target_age = child.current_age
                        self.logger.info(f"🎯 Age GROUNDED from child profile: {child.name} is {effective_target_age} years old (born {child.birth_year})")

                    # Fetch parent account for language
                    if language is None:
                        parent = await self.storage.get_parent_account(child.parent_id)
                        if parent:
                            effective_language = parent.language
                            self.logger.info(f"🌍 Language GROUNDED from parent: {effective_language}")

                    self.logger.info(f"📋 Story initialized for {child.name} (age {effective_target_age}, language: {effective_language})")
                else:
                    self.logger.warning(f"⚠️ Child profile '{child_id}' not found - using default age {effective_target_age}")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not fetch child profile {child_id}: {e} - using default age {effective_target_age}")

        # Create story object
        from src.models import StoryPreferences

        story_prefs = StoryPreferences(**(preferences or {}))

        # Override language in preferences if derived from parent
        if effective_language:
            story_prefs.language = effective_language

        # Set target_age in preferences (from child profile or parameter)
        story_prefs.target_age = effective_target_age

        story = Story(
            prompt=prompt,
            preferences=story_prefs,
            status=StoryStatus.INITIALIZING
        )

        # Update metadata with both child_id and user_id (for migration compatibility)
        story.metadata.child_id = effective_child_id
        story.metadata.user_id = effective_child_id  # Keep for backward compat

        # Save initial story to Firebase
        await self.storage.save_story(story)

        # Add story to child's story_ids if using child profile
        if child_id:
            try:
                await self.storage.add_story_to_child(child_id, story.id)
                await self.storage.set_active_story(child_id, story.id)
            except Exception as e:
                self.logger.warning(f"Could not add story to child profile: {e}")

        self.logger.job_received("initialize_story", story.id, f"Prompt: {prompt[:50]}...")

        # Story initialization - welcome message already handled by CompanionAgent
        # via /conversation/start endpoint (conversation-first UX)
        self.logger.info(f"Story initialized: {story.id} (greeting handled by CompanionAgent)")

        # Update story status
        story.status = StoryStatus.IN_DIALOGUE
        await self.storage.save_story(story)

        # Emit story_created event (no welcome message - already sent by CompanionAgent)
        await story_events.emit(
            "story_created",
            story.id,
            {
                "story_id": story.id,
                "status": story.status.value,
                "prompt": prompt
            }
        )
        print(f"📡 Emitted story_created event for {story.id}")

        self.logger.job_completed("initialize_story", story.id, time.time() - start_time)

        return {
            "success": True,
            "story_id": story.id,
            "status": story.status.value,
            "story": story.model_dump()
            # No welcome_message - handled by CompanionAgent via /conversation/start
        }

    # Phase 2 enhancements planned - see docs/ROADMAP.md

    async def generate_story_structure(
        self,
        story_id: str,
        dialogue_context: str = ""
    ) -> Dict[str, Any]:
        """
        Generate complete story structure (background task).

        Args:
            story_id: Story ID
            dialogue_context: Context from user dialogue

        Returns:
            Dict with structure and characters
        """
        import time
        structure_start_time = time.time()

        # Get story from Firebase
        story = await self.storage.get_story(story_id)
        if not story:
            return {"success": False, "error": "Story not found"}

        # === OBSERVATORY: Emit pipeline and agent events ===
        await story_events.emit_pipeline_stage(story_id, "structure", "in_progress")
        await story_events.emit_agent_started(
            story_id, "StructureAgent", "Planning story outline and characters"
        )

        # Update status
        story.status = StoryStatus.STRUCTURE_READY
        await self.storage.update_story_status(story_id, story.status.value)

        # Determine story complexity for character planning
        themes_list = story.preferences.themes or []
        themes_str = ', '.join(themes_list) if themes_list else 'adventure'
        is_historical = story.preferences.educational_focus == 'history' or any(t in ['history', 'historical'] for t in themes_list)

        # Get target_age from preferences (grounded from child profile)
        target_age = story.preferences.target_age if story.preferences.target_age else 10

        # Task 2: Create story structure with dynamic character cast
        structure_prompt = get_generate_structure_prompt(
            story_prompt=story.prompt,
            difficulty=story.preferences.difficulty.value,
            target_age=target_age,
            educational_focus=story.preferences.educational_focus or 'general knowledge',
            themes=themes_str,
            scary_level=story.preferences.scary_level,
            dialogue_context=dialogue_context,
            user_style_requests=self._format_user_style_requests(story.preferences.user_style_requests),
            is_historical=is_historical
        )

        structure_task = Task(
            description=structure_prompt,
            agent=self.structure_agent,
            expected_output="Valid JSON story structure with character arcs"
        )

        # Execute structure creation (in separate thread to avoid blocking event loop)
        crew = Crew(
            agents=[self.structure_agent],
            tasks=[structure_task],
            process=Process.sequential,
            verbose=True
        )

        # Set up model tracking for StructureAgent
        tracker = ModelTracker.get_instance()
        configured_model = self.router.get_model_for_agent("structure")
        tracker.set_current_agent("StructureAgent", configured_model=configured_model)

        structure_result = await self._run_crew_async(crew)

        # Get actual model used from tracker
        structure_model = tracker.get_last_model("StructureAgent")

        # LOG: Raw output from agent
        raw_output = str(structure_result).strip()
        self.logger.info(f"📤 StructureAgent raw output length: {len(raw_output)} chars")
        self.logger.debug("StructureAgent", f"First 500 chars: {raw_output[:500]}")
        self.logger.debug("StructureAgent", f"Last 500 chars: {raw_output[-500:]}")

        # Parse and save structure
        try:
            # Use the full _clean_json_output method which handles:
            # 1. Markdown code block extraction
            # 2. Control character sanitization
            # 3. Unescaped quote repair (e.g., "Harald Tanglehair" -> \"Harald Tanglehair\")
            result_str = clean_json_output(raw_output)

            self.logger.debug("StructureAgent", f"Cleaned JSON length: {len(result_str)} chars")
            self.logger.debug("StructureAgent", "Attempting to parse JSON...")

            structure_data = json.loads(result_str)
            self.logger.info(f"   ✅ Successfully parsed structure JSON")
            self.logger.debug("StructureAgent", f"Structure contains: {len(structure_data.get('chapters', []))} chapters, {len(structure_data.get('characters_needed', []))} characters")

            await self.storage.save_structure(story_id, structure_data)
            self.logger.info(f"   💾 Structure saved to Firebase")

            # Also update story object
            story.structure = StoryStructure(**structure_data)

            # === NARRATIVE METHOD PLANNING (Guillermo + Stephen Debate) ===
            # Determine HOW the story should be told (Harry Potter linear vs Da Vinci Code multi-POV)
            self.logger.info("📖 Planning narrative method (Guillermo + Stephen debate)...")
            narrative_method = await self._plan_narrative_method(story, structure_data)
            story.structure.narrative_method = narrative_method
            self.logger.info(f"   ✅ Narrative method set: {narrative_method.method}")

            await self.storage.save_story(story)

            # Emit structure_ready event with data
            event_data = {
                "title": structure_data.get("title"),
                "chapters": len(structure_data.get("chapters", [])),
                "characters_needed": len(structure_data.get("characters_needed", [])),
                "narrative_method": narrative_method.method,
                "pov_characters": narrative_method.pov_characters
            }
            await story_events.emit("structure_ready", story_id, event_data)
            # NOTE: CompanionAgent handles user communication for structure_ready

            # === OBSERVATORY: Emit completion events ===
            duration_ms = int((time.time() - structure_start_time) * 1000)
            await story_events.emit_agent_completed(
                story_id, "StructureAgent", duration_ms, True,
                f"Planned {len(structure_data.get('chapters', []))} chapters, {len(structure_data.get('characters_needed', []))} characters. Narrative: {narrative_method.method}",
                model=structure_model  # Actual model (e.g., "gpt-oss-120b")
            )
            await story_events.emit_pipeline_stage(story_id, "structure", "completed", details=event_data)

            return {
                "success": True,
                "structure": structure_data
            }

        except json.JSONDecodeError as e:
            self.logger.error("StructureAgent", f"JSON parsing failed: {e}", e)
            self.logger.error("StructureAgent", f"Error position: line {e.lineno}, column {e.colno}")
            self.logger.error("StructureAgent", f"Problem area: ...{raw_output[max(0, e.pos-100):e.pos+100]}...")

            # Save raw output to file for debugging
            debug_file = Path("logs/debug") / f"failed_structure_{story_id}.txt"
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            debug_file.write_text(raw_output, encoding='utf-8')
            self.logger.error("StructureAgent", f"Raw output saved to: {debug_file}")

            return {
                "success": False,
                "error": f"Failed to parse structure: {e}",
                "raw_output": raw_output[:1000] + "..." if len(raw_output) > 1000 else raw_output
            }

    # =========================================================================
    # STRUCTURE V2: Character-Aware Refinement
    # =========================================================================

    async def refine_structure_v2(self, story_id: str) -> Dict[str, Any]:
        """
        Refine story structure after Chapter 1 is written.

        Called when Chapter 1 playback begins. Uses actual Chapter 1 content,
        D&D character cards with skills, and user dialogue inputs to rewrite
        synopses for chapters 2-N.

        Uses Azure AI Foundry Model Router for unified model access and A/B testing.
        This runs in background (non-blocking) and is non-fatal if it fails.
        The original structure is preserved as fallback.

        Args:
            story_id: Story ID

        Returns:
            Dict with:
                - success: bool
                - refined_chapters: int (count of chapters refined)
                - error: str (if failed)
                - original_preserved: bool (True if using fallback)
                - model: str (actual model used via Foundry)
        """
        import litellm
        from src.models.models import StructureRefinementV2, ChapterOutline

        refinement_start = time.time()
        self.logger.info(f"📚 Structure V2: Starting refinement for {story_id}")

        # Get the configured model for StructureAgent from LLM Router
        # This respects TEST_STRUCTURE_MODEL env var overrides for A/B testing
        structure_model = self.router.get_model_for_agent("structure")
        structure_config = self.router.get_llm_kwargs("structure")
        self.logger.info(f"📚 Structure V2: Using model {structure_model} (from LLM Router)")

        try:
            # === GATHER INPUTS ===
            story = await self.storage.get_story(story_id)
            if not story or not story.structure:
                return {"success": False, "error": "Story or structure not found"}

            # Check if already refined (avoid re-running)
            if hasattr(story.structure, 'refinement_v2') and story.structure.refinement_v2 is not None:
                self.logger.info(f"📚 Structure V2: Already refined for {story_id}, skipping")
                return {"success": True, "refined_chapters": 0, "message": "Already refined", "skipped": True}

            # Check we have chapters to refine (need at least 2 chapters)
            if len(story.structure.chapters) < 2:
                self.logger.info(f"📚 Structure V2: Only 1 chapter, nothing to refine")
                return {"success": True, "refined_chapters": 0, "message": "Single chapter story"}

            # Get Chapter 1 content
            chapter_1 = next((ch for ch in story.chapters if ch.number == 1), None)
            if not chapter_1 or not chapter_1.content:
                return {"success": False, "error": "Chapter 1 not written yet"}

            # Get all character cards
            characters = await self.storage.get_characters(story_id)
            if not characters:
                self.logger.warning(f"📚 Structure V2: No characters found, skipping refinement")
                return {"success": False, "error": "No characters found", "original_preserved": True}

            # Get user inputs from dialogue phase
            reading_state = await self.storage.get_reading_state(story_id)
            user_inputs = []
            if reading_state and hasattr(reading_state, 'queued_inputs'):
                user_inputs = reading_state.queued_inputs or []

            # === EMIT OBSERVATORY EVENTS ===
            await story_events.emit_agent_started(
                story_id, "StructureAgent", "V2: Refining synopses with character skills"
            )
            await story_events.emit_pipeline_stage(
                story_id, "structure_v2", "in_progress",
                details={"trigger": "chapter_1_playback"}
            )

            # === BUILD CONTEXT ===
            # Format character cards for the prompt
            character_cards = []
            skills_available = []
            for char in characters:
                card = {
                    "name": char.name,
                    "role": char.role,
                    "personality_traits": char.personality_traits,
                    "skills": []
                }
                if char.progression and char.progression.skills_learned:
                    for skill in char.progression.skills_learned:
                        skill_info = f"{skill.name} (Level {skill.level})"
                        card["skills"].append(skill_info)
                        skills_available.append(skill_info)
                if char.character_arc:
                    card["arc"] = char.character_arc
                character_cards.append(card)

            # Format user inputs
            user_prefs_summary = "None specified"
            if user_inputs:
                prefs = [inp.raw_input for inp in user_inputs[:5]]  # Top 5
                user_prefs_summary = "; ".join(prefs)

            # Get original synopses for chapters 2-N
            original_synopses = []
            for ch in story.structure.chapters[1:]:  # Skip Chapter 1
                original_synopses.append({
                    "number": ch.number,
                    "title": ch.title,
                    "synopsis": ch.synopsis,
                    "characters_featured": ch.characters_featured,
                    "educational_points": ch.educational_points,
                    "character_development_milestones": ch.character_development_milestones
                })

            # === BUILD PROMPT ===
            prompt = get_refine_structure_prompt(
                story_title=story.structure.title,
                story_theme=story.structure.theme,
                educational_focus=story.preferences.educational_focus if story.preferences else 'general',
                total_chapters=len(story.structure.chapters),
                chapter_1_content=chapter_1.content,
                character_cards=character_cards,
                user_prefs_summary=user_prefs_summary,
                original_synopses=original_synopses
            )

            # === EXECUTE REFINEMENT VIA LITELLM ===
            # Uses the model configured in LLM Router (respects TEST_STRUCTURE_MODEL)
            limiter = get_rate_limiter()
            try:
                async with limiter.acquire(structure_model):
                    self.logger.info(f"📚 Structure V2: Calling LiteLLM with model={structure_model}")

                    response = await asyncio.wait_for(
                        litellm.acompletion(
                            model=structure_model,
                            messages=[
                                {"role": "system", "content": get_refine_structure_system_prompt()},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=structure_config.get("max_tokens", 8000),
                            temperature=0.6,  # More constrained than initial creative pass
                            timeout=structure_config.get("timeout", 180),
                            drop_params=True  # Auto-drop unsupported params per provider
                        ),
                        timeout=180.0  # 3 minute timeout (increased for longer output)
                    )

                    # Add delay after API call to respect rate limits
                    await asyncio.sleep(self._foundry_delay)

            except asyncio.TimeoutError:
                self.logger.warning(f"📚 Structure V2: Timeout for {story_id}, using original")
                await story_events.emit_pipeline_stage(
                    story_id, "structure_v2", "timeout",
                    details={"fallback": "original_preserved"}
                )
                return {"success": True, "original_preserved": True, "error": "Timeout"}

            # === PARSE RESULT ===
            # LiteLLM returns OpenAI-style response object
            raw_output = response.choices[0].message.content.strip()
            actual_model = response.model or structure_model
            self.logger.info(f"📚 Structure V2: Got {len(raw_output)} chars from {actual_model}")

            try:
                result_str = clean_json_output(raw_output)
                refinement_data = json.loads(result_str)
            except json.JSONDecodeError as e:
                self.logger.error("StructureV2", f"JSON parse failed: {e}")
                return {"success": False, "error": f"JSON parse failed: {e}", "original_preserved": True}

            # === APPLY REFINEMENT ===
            refined_chapters = refinement_data.get("refined_chapters", [])
            if not refined_chapters:
                return {"success": False, "error": "No refined chapters in output", "original_preserved": True}

            # Build new chapters list: keep Chapter 1, replace rest
            new_chapters = [story.structure.chapters[0]]  # Keep original Chapter 1

            for refined in refined_chapters:
                try:
                    # Create ChapterOutline from refined data
                    chapter_outline = ChapterOutline(
                        number=refined["number"],
                        title=refined.get("title", f"Chapter {refined['number']}"),
                        synopsis=refined["synopsis"],
                        characters_featured=refined.get("characters_featured", []),
                        educational_points=refined.get("educational_points", []),
                        facts_to_verify=refined.get("facts_to_verify", []),
                        character_development_milestones=refined.get("character_development_milestones", {})
                    )
                    new_chapters.append(chapter_outline)
                except Exception as e:
                    self.logger.warning(f"📚 Structure V2: Failed to parse chapter {refined.get('number')}: {e}")
                    # Use original chapter as fallback
                    original_ch = next((ch for ch in story.structure.chapters if ch.number == refined.get("number")), None)
                    if original_ch:
                        new_chapters.append(original_ch)

            # Update story structure
            story.structure.chapters = new_chapters
            story.structure.refinement_v2 = StructureRefinementV2(
                chapters_refined=len(refined_chapters),
                user_inputs_incorporated=len(user_inputs),
                skills_leveraged=refinement_data.get("skills_leveraged", []),
                notes=refinement_data.get("refinement_notes", "")
            )

            # Save to Firebase
            await self.storage.save_structure(story_id, story.structure.model_dump(mode='json'))
            await self.storage.save_story(story)

            # === EMIT COMPLETION ===
            duration_ms = int((time.time() - refinement_start) * 1000)
            skills_count = len(refinement_data.get("skills_leveraged", []))

            await story_events.emit_agent_completed(
                story_id, "StructureAgent", duration_ms, True,
                f"V2: Refined {len(refined_chapters)} chapters, leveraged {skills_count} skills",
                model=actual_model  # Actual model from Foundry router (e.g., "gpt-4o", "claude-sonnet-4")
            )
            await story_events.emit_pipeline_stage(
                story_id, "structure_v2", "completed",
                details={"chapters_refined": len(refined_chapters), "skills_leveraged": skills_count}
            )

            self.logger.info(f"✅ Structure V2: Refined {len(refined_chapters)} chapters for {story_id} (model: {actual_model})")

            return {
                "success": True,
                "refined_chapters": len(refined_chapters),
                "skills_leveraged": refinement_data.get("skills_leveraged", []),
                "original_preserved": False,
                "model": actual_model
            }

        except Exception as e:
            self.logger.error("StructureV2", f"Failed for {story_id}: {e}", e)
            await story_events.emit_pipeline_stage(
                story_id, "structure_v2", "error",
                details={"error": str(e), "fallback": "original_preserved"}
            )
            # Non-fatal - original structure remains valid
            return {
                "success": False,
                "error": str(e),
                "original_preserved": True
            }

    async def create_characters(
        self,
        story_id: str
    ) -> Dict[str, Any]:
        """
        Create all characters for the story IN PARALLEL.

        Uses asyncio.gather() to create multiple characters simultaneously,
        reducing character creation time from ~5 minutes to ~1.5 minutes.

        Args:
            story_id: Story ID

        Returns:
            Dict with created characters
        """
        import asyncio

        import time
        characters_start_time = time.time()

        story = await self.storage.get_story(story_id)
        if not story or not story.structure:
            return {"success": False, "error": "Story structure not found"}

        # === OBSERVATORY: Emit pipeline stage ===
        total_chars = len(story.structure.characters_needed) if story.structure.characters_needed else 0
        await story_events.emit_pipeline_stage(story_id, "characters", "in_progress", f"0/{total_chars}")

        # FIX: Create ALL characters upfront for complete stories
        # The "major only" filter caused incomplete character rosters (1/8 characters)
        # Better to wait longer for complete character generation

        # === Fetch existing characters for deduplication ===
        # NOTE: We allow new characters to be added even if some exist
        # Round Table or agents may suggest additional characters during story generation
        existing_characters = await self.storage.get_characters(story_id)

        # === SMART DEDUPLICATION: Track existing names for partial creation ===
        # This handles cases where some but not all characters exist
        existing_char_names = set()
        existing_char_summaries = []  # For passing to CharacterAgent

        if existing_characters:
            for char in existing_characters:
                char_name = char.name if hasattr(char, 'name') else char.get('name', '')
                existing_char_names.add(char_name.lower().strip())
                # Also track variations (first name, last name, nicknames)
                for part in char_name.split():
                    if len(part) > 2:  # Skip short words like "the", "of"
                        existing_char_names.add(part.lower().strip())
                # Build summary for CharacterAgent to detect evolutions
                existing_char_summaries.append({
                    "name": char_name,
                    "role": char.role if hasattr(char, 'role') else char.get('role', ''),
                    "traits": (char.personality_traits if hasattr(char, 'personality_traits')
                              else char.get('personality_traits', []))[:3]  # First 3 traits
                })
            self.logger.info(f"📋 Found {len(existing_characters)} existing characters in Firebase")

        # Deduplicate characters by name to avoid creating the same character twice
        # StructureAgent sometimes outputs duplicate entries in characters_needed
        seen_names = set()
        characters_to_create = []
        for char_spec in story.structure.characters_needed:
            # Extract name from either dict or CharacterNeeded object
            if isinstance(char_spec, dict):
                name = char_spec.get('name', '')
            elif hasattr(char_spec, 'name'):
                name = char_spec.name
            else:
                name = str(char_spec)

            name_lower = name.lower().strip()

            # Check 1: Already seen in this batch (StructureAgent duplicates)
            if name and name_lower in seen_names:
                self.logger.info(f"   ⏭️ Skipping batch duplicate: {name}")
                continue

            # Check 2: Already exists in Firebase (exact match or partial match)
            if name_lower in existing_char_names:
                self.logger.info(f"   ⏭️ Skipping - already in Firebase: {name}")
                continue

            # Check 3: Name part match (e.g., "Harald Fairhair" matches existing "Harald Hårfagre")
            name_parts = [p.lower().strip() for p in name.split() if len(p) > 2]
            if any(part in existing_char_names for part in name_parts):
                matching_part = next(p for p in name_parts if p in existing_char_names)
                self.logger.info(f"   ⏭️ Skipping - likely duplicate ('{matching_part}' exists): {name}")
                continue

            # This character is new - add to creation list
            seen_names.add(name_lower)
            characters_to_create.append(char_spec)

        self.logger.info(f"📊 Creating ALL {len(characters_to_create)} unique characters for complete story (deduped from {len(story.structure.characters_needed)})")

        # ===== SMART BATCHING: Group by importance, parallelize within batches =====
        # This maintains data consistency (major characters created before supporting)
        # while parallelizing within each importance tier using AdaptiveRateLimiter

        def get_importance(char_spec) -> str:
            """Extract importance level from character spec."""
            if isinstance(char_spec, dict):
                return char_spec.get('importance', 'supporting').lower()
            return getattr(char_spec, 'importance', 'supporting').lower()

        def get_char_name(char_spec, fallback: str = 'Unknown') -> str:
            """Extract character name from spec."""
            if isinstance(char_spec, dict):
                return char_spec.get('name', fallback)
            return getattr(char_spec, 'name', fallback)

        # Group characters by importance tier
        importance_batches = {
            'major': [],      # Protagonists, main characters - create first
            'supporting': [], # Secondary characters - create second
            'minor': []       # Minor, thematic, cameo, etc. - create last
        }

        for char_spec in characters_to_create:
            importance = get_importance(char_spec)
            if importance in ['major', 'protagonist', 'antagonist', 'main']:
                importance_batches['major'].append(char_spec)
            elif importance in ['supporting', 'secondary']:
                importance_batches['supporting'].append(char_spec)
            else:
                importance_batches['minor'].append(char_spec)

        self.logger.info(f"🚀 Starting SMART BATCHED character generation:")
        self.logger.info(f"   📌 Major: {len(importance_batches['major'])} characters")
        self.logger.info(f"   📌 Supporting: {len(importance_batches['supporting'])} characters")
        self.logger.info(f"   📌 Minor: {len(importance_batches['minor'])} characters")

        characters_created = []
        failed_characters = []
        global_index = 0  # Track position across all batches
        limiter = get_rate_limiter()

        async def create_character_with_rate_limit(
            char_spec,
            index: int,
            existing_char_summaries: List[Dict],
            existing_character_objects: List
        ) -> Tuple[int, Any, Optional[str]]:
            """Create a single character with rate limiting and retry."""
            char_name = get_char_name(char_spec, f'Character {index+1}')
            max_retries = 3

            # Set up model tracking
            tracker = ModelTracker.get_instance()
            configured_model = self.router.get_model_for_agent("character")
            tracker.set_current_agent(f"CharacterAgent_{index}", configured_model=configured_model)

            # Emit agent_started event
            await story_events.emit("agent_started", story_id, {
                "agent": "CharacterAgent",
                "task": f"Creating {char_name}",
                "instance": index + 1,
                "total_instances": len(characters_to_create)
            })

            result = None
            error_message = None

            for attempt in range(max_retries):
                try:
                    async with limiter.acquire("azure"):
                        result = await self._create_single_character(
                            story_id, story, char_spec,
                            existing_characters=existing_char_summaries,
                            existing_character_objects=existing_character_objects
                        )
                        break  # Success, exit retry loop
                except RateLimitError:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"🔄 {char_name}: Retry {attempt + 1}/{max_retries} after rate limit")
                        continue
                    error_message = f"Rate limit exceeded after {max_retries} retries"
                    self.logger.error(f"❌ {char_name}: {error_message}")
                except Exception as e:
                    error_message = str(e)[:100]
                    self.logger.error("CharacterAgent", f"Character creation failed for {char_name}: {error_message}")
                    break  # Don't retry on non-rate-limit errors

            # Get actual model used
            actual_model = tracker.get_last_model(f"CharacterAgent_{index}")

            # Emit agent_completed event
            event_data = {
                "agent": "CharacterAgent",
                "task": f"Created {char_name}" if result else f"Failed: {char_name}",
                "instance": index + 1,
                "total_instances": len(characters_to_create),
                "success": result is not None,
                "model": actual_model
            }
            if error_message:
                event_data["error"] = error_message
            await story_events.emit("agent_completed", story_id, event_data)

            return (index, result, error_message if not result else None)

        async def process_semantic_batch_sequentially(
            semantic_batch: List,
            batch_idx: int,
            total_batches: int,
            base_index: int
        ) -> List[Tuple[int, Any, Optional[str]]]:
            """
            Process a single semantic batch SEQUENTIALLY.

            Each character sees all previously created characters (full visibility),
            including those created earlier in this same batch.

            This prevents duplicate creation of similar characters like
            "Harald Fairhair" and "Harald the Bold".

            Args:
                semantic_batch: List of character specs in this batch
                batch_idx: Index of this batch (for logging)
                total_batches: Total number of semantic batches
                base_index: Starting global index for this batch

            Returns:
                List of (index, result, error) tuples
            """
            results = []

            for char_idx, char_spec in enumerate(semantic_batch):
                char_name = get_char_name(char_spec, f'Character {base_index + char_idx + 1}')

                # Fetch FRESH existing characters each time
                # This includes characters just created in this batch
                fresh_existing = await self.storage.get_characters(story_id)
                fresh_summaries = []
                if fresh_existing:
                    for char in fresh_existing:
                        c_name = char.name if hasattr(char, 'name') else char.get('name', '')
                        c_role = char.role if hasattr(char, 'role') else char.get('role', '')
                        c_traits = (char.personality_traits if hasattr(char, 'personality_traits')
                                   else char.get('personality_traits', []))[:3]
                        fresh_summaries.append({"name": c_name, "role": c_role, "traits": c_traits})

                self.logger.info(f"      ➡️ Batch {batch_idx+1}/{total_batches}: Creating {char_name} ({char_idx+1}/{len(semantic_batch)}) - sees {len(fresh_summaries)} existing chars")

                # Create with rate limiting
                result = await create_character_with_rate_limit(
                    char_spec,
                    base_index + char_idx,
                    fresh_summaries,
                    fresh_existing or []
                )

                results.append(result)

            return results

        # Process each importance tier sequentially, but parallelize within each tier
        batch_order = ['major', 'supporting', 'minor']

        for batch_name in batch_order:
            importance_batch = importance_batches[batch_name]
            if not importance_batch:
                continue

            # === SEMANTIC BATCHING ===
            # Split importance batch into semantic batches where similar names go to DIFFERENT batches
            # This ensures sequential processing within batches catches potential duplicates
            from src.services.character_service import assign_to_semantic_batches
            semantic_batches = assign_to_semantic_batches(importance_batch, max_batch_size=4, get_name=get_char_name)

            self.logger.info(f"\n   🎭 Processing {batch_name.upper()}: {len(importance_batch)} chars → {len(semantic_batches)} semantic batches")
            for i, sb in enumerate(semantic_batches):
                names = [get_char_name(c) for c in sb]
                self.logger.info(f"      📦 Batch {i+1}: [{', '.join(names)}]")

            # Process semantic batches in PARALLEL, but SEQUENTIAL within each batch
            # Calculate base indices for each semantic batch
            batch_base_indices = []
            running_idx = global_index
            for sb in semantic_batches:
                batch_base_indices.append(running_idx)
                running_idx += len(sb)

            semantic_tasks = [
                process_semantic_batch_sequentially(
                    semantic_batch,
                    batch_idx,
                    len(semantic_batches),
                    batch_base_indices[batch_idx]
                )
                for batch_idx, semantic_batch in enumerate(semantic_batches)
            ]

            # Execute all semantic batches with timeout
            try:
                all_batch_results = await asyncio.wait_for(
                    asyncio.gather(*semantic_tasks, return_exceptions=True),
                    timeout=300  # 5 minute timeout per importance tier
                )

                # Flatten and process results from all semantic batches
                flat_char_idx = 0
                for batch_idx, batch_results in enumerate(all_batch_results):
                    semantic_batch = semantic_batches[batch_idx]

                    if isinstance(batch_results, Exception):
                        self.logger.error("CharacterAgent", f"Semantic batch {batch_idx+1} failed: {batch_results}")
                        # Mark all characters in this batch as failed
                        for i, char_spec in enumerate(semantic_batch):
                            failed_characters.append((global_index + flat_char_idx + i, char_spec))
                        flat_char_idx += len(semantic_batch)
                        continue

                    # Process individual results from this semantic batch
                    for result_idx, result in enumerate(batch_results):
                        char_spec = semantic_batch[result_idx]
                        char_name = get_char_name(char_spec)

                        if isinstance(result, tuple):
                            idx, char_result, error = result
                            if char_result is not None:
                                characters_created.append(char_result)
                                self.logger.info(f"      ✅ {char_name}")
                            else:
                                failed_characters.append((idx, char_spec))
                                self.logger.warning(f"      ❌ {char_name}: {error or 'Unknown error'}")
                        else:
                            self.logger.warning(f"      ⚠️ Unexpected result type for {char_name}: {type(result)}")
                            failed_characters.append((global_index + flat_char_idx + result_idx, char_spec))

                    flat_char_idx += len(semantic_batch)

                # Emit progress after importance tier completes
                await story_events.emit_pipeline_stage(
                    story_id, "characters", "in_progress",
                    f"{len(characters_created)}/{total_chars}"
                )

            except asyncio.TimeoutError:
                self.logger.error("CharacterAgent", f"{batch_name.upper()} tier timeout after 300s")
                # Mark remaining characters as failed
                for i, char_spec in enumerate(importance_batch):
                    if (global_index + i) not in [idx for idx, _ in failed_characters]:
                        failed_characters.append((global_index + i, char_spec))

            global_index += len(importance_batch)
            succeeded_count = len([c for c in characters_created]) - (global_index - len(importance_batch))
            self.logger.info(f"      ✅ {batch_name.upper()} tier complete: {len(importance_batch)} characters processed")

        # RETRY FAILED CHARACTERS (up to 2 retry rounds with increasing delays)
        max_retry_rounds = 2
        retry_delays = [5, 10]  # 5 seconds for first retry, 10 for second

        for retry_round in range(max_retry_rounds):
            if not failed_characters:
                break

            delay = retry_delays[retry_round] if retry_round < len(retry_delays) else 15
            self.logger.info(f"🔄 Retry round {retry_round + 1}/{max_retry_rounds}: {len(failed_characters)} failed characters (waiting {delay}s)...")
            await asyncio.sleep(delay)

            still_failed = []

            # Fetch fresh existing characters for retry batch
            retry_existing_chars = await self.storage.get_characters(story_id)
            retry_char_summaries = []
            if retry_existing_chars:
                for char in retry_existing_chars:
                    c_name = char.name if hasattr(char, 'name') else char.get('name', '')
                    c_role = char.role if hasattr(char, 'role') else char.get('role', '')
                    c_traits = (char.personality_traits if hasattr(char, 'personality_traits')
                               else char.get('personality_traits', []))[:3]
                    retry_char_summaries.append({"name": c_name, "role": c_role, "traits": c_traits})

            # Retry all failed characters in parallel
            retry_tasks = [
                create_character_with_rate_limit(char_spec, idx, retry_char_summaries, retry_existing_chars or [])
                for idx, char_spec in failed_characters
            ]

            try:
                retry_results = await asyncio.wait_for(
                    asyncio.gather(*retry_tasks, return_exceptions=True),
                    timeout=180  # 3 minute timeout for retries
                )

                for i, result in enumerate(retry_results):
                    idx, char_spec = failed_characters[i]
                    char_name = get_char_name(char_spec)

                    if isinstance(result, Exception):
                        still_failed.append((idx, char_spec))
                    elif isinstance(result, tuple):
                        _, char_result, error = result
                        if char_result is not None:
                            characters_created.append(char_result)
                            self.logger.info(f"   ✅ Retry {retry_round + 1} succeeded: {char_name}")
                        else:
                            still_failed.append((idx, char_spec))
                    else:
                        still_failed.append((idx, char_spec))

            except asyncio.TimeoutError:
                self.logger.error(f"⏱️ Retry round {retry_round + 1} timeout")
                still_failed = failed_characters  # All remaining are still failed

            failed_characters = still_failed

        # Log final failures
        if failed_characters:
            failed_names = [
                get_char_name(char_spec, 'Unknown')
                for _, char_spec in failed_characters
            ]
            self.logger.warning(f"⚠️ Characters that failed after all retries: {', '.join(failed_names)}")

        expected_count = len(characters_to_create)
        created_count = len(characters_created)

        self.logger.info(f"✅ Character generation complete: {created_count}/{expected_count} characters created")

        # === OBSERVATORY: Emit completion events ===
        duration_ms = int((time.time() - characters_start_time) * 1000)

        if created_count == 0:
            self.logger.error("CharacterAgent", f"CRITICAL: No characters were successfully created (attempted {expected_count})")
            await story_events.emit_pipeline_stage(story_id, "characters", "error", f"0/{expected_count}")
            return {
                "success": False,
                "error": f"Failed to create any characters. Attempted {expected_count}, created 0.",
                "characters": []
            }

        if created_count < expected_count:
            self.logger.warning(f"⚠️  Only {created_count}/{expected_count} characters created (after retry)")

        # Emit success events
        await story_events.emit_pipeline_stage(
            story_id, "characters", "completed", f"{created_count}/{expected_count}",
            {"characters_created": [c.get("name", "Unknown") for c in characters_created]}
        )

        # === BILL LLM DEDUPLICATION PASS ===
        # After all characters are created, run Bill to identify any remaining duplicates
        # that the rule-based matching might have missed
        final_characters = await self.storage.get_characters(story_id)
        if len(final_characters) > expected_count:
            self.logger.info(f"📊 More characters than expected ({len(final_characters)} vs {expected_count}) - running Bill dedup...")

            # Build story context for Bill
            story_title = story.structure.title if story.structure else "Unknown Story"
            story_theme = story.structure.theme if story.structure and hasattr(story.structure, 'theme') else ""
            story_context = f"{story_title} - {story_theme}" if story_theme else story_title

            # Ask Bill to identify duplicates (via CharacterService)
            duplicates = await self.character_service.dedupe_characters_with_llm(
                story_id, final_characters, story_context
            )

            # Merge any duplicates found (via CharacterService)
            merged_count = 0
            for duplicate_name, original_name in duplicates:
                success = await self.character_service.merge_characters(
                    story_id, duplicate_name, original_name, final_characters
                )
                if success:
                    merged_count += 1

            if merged_count > 0:
                self.logger.info(f"✅ Bill merged {merged_count} duplicate characters")
                # Refresh character count
                final_characters = await self.storage.get_characters(story_id)
                created_count = len(final_characters)

        return {
            "success": True,
            "characters": characters_created,
            "expected_count": expected_count,
            "created_count": created_count
        }

    async def _create_single_character(
        self,
        story_id: str,
        story,
        char_spec,
        existing_characters: list = None,
        existing_character_objects: list = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a single character. Called in parallel by create_characters().

        Args:
            story_id: Story ID
            story: Story object
            char_spec: Character specification (dict or CharacterNeeded)
            existing_characters: List of existing character summaries for dedupe check (for LLM prompt)
            existing_character_objects: Actual Character objects for identity matching

        Returns the character dict on success, or None on failure.
        """
        from crewai import Task, Crew, Process
        from src.services.events import story_events

        # Handle both old format (strings) and new format (objects)
        if isinstance(char_spec, str):
            char_name = char_spec
            char_role = "supporting"
            char_importance = "supporting"
            arc_milestones = None
        else:
            # Handle both dict and CharacterNeeded object
            if isinstance(char_spec, dict):
                char_name = char_spec.get('name')
                if not char_name:
                    self.logger.warning(f"CharacterAgent: Character spec missing 'name': {char_spec}")
                    return None
                char_role = char_spec.get('role', 'supporting')
                char_importance = char_spec.get('importance', 'supporting')
                arc_milestones = char_spec.get('arc_milestones', None)
            else:
                # CharacterNeeded object
                char_name = char_spec.name
                char_role = char_spec.role
                char_importance = char_spec.importance
                arc_milestones = char_spec.arc_milestones

        # === IDENTITY MATCHING: Check if this is a variant of an existing character ===
        if existing_character_objects:
            self.logger.info(f"🔍 Checking identity for '{char_name}' against {len(existing_character_objects)} existing characters")
            for ec in existing_character_objects[:5]:  # Log first 5 for debugging
                self.logger.debug("CharacterMatch", f"   - Existing: '{ec.name}'")
            matched_char = match_character_identity(char_name, existing_character_objects)
            if matched_char:
                self.logger.info(
                    f"📝 Character '{char_name}' matches existing '{matched_char.name}' - skipping creation"
                )
                # Return the existing character data (as dict) instead of creating new
                return {
                    "MATCHED_EXISTING": matched_char.name,
                    "character_id": matched_char.id,
                    "name": matched_char.name,
                    "role": matched_char.role,
                    "skipped": True
                }

        # Build arc description
        arc_description = ""
        if arc_milestones:
            arc_description = f"""
            CHARACTER ARC MILESTONES:
            This is a {char_importance} character with planned development:
            {json.dumps(arc_milestones, indent=2)}

            Give this character initial skills (2-3 at level 1-3) that will grow.
            Choose personality traits that align with their arc journey.
            """

        # Determine skill requirements based on importance
        if char_importance == "major":
            skill_requirement = "3-5 skills at levels 1-4"
            min_skills = 3
        elif char_importance == "supporting":
            skill_requirement = "2-3 skills at levels 1-3"
            min_skills = 2
        else:
            skill_requirement = "1-2 skills at levels 1-2"
            min_skills = 1

        # Build existing characters context for dedupe check
        existing_chars_context = ""
        if existing_characters and len(existing_characters) > 0:
            chars_list = "\n".join([
                f"   - {c['name']} ({c['role']}): {', '.join(c['traits'][:2]) if c['traits'] else 'no traits'}"
                for c in existing_characters
            ])
            existing_chars_context = f"""
            === EXISTING CHARACTERS (DO NOT DUPLICATE) ===
            These characters already exist in this story:
{chars_list}

            ⚠️ CRITICAL DEDUPE CHECK:
            Before creating "{char_name}", verify this is NOT a duplicate of any existing character.
            Check for:
            1. SAME NAME with different spelling (e.g., "Harald Fairhair" = "Harald Hårfagre")
            2. SAME ROLE with different name (e.g., "the shipwright" already exists)
            3. CHARACTER EVOLUTION (e.g., "the bad wolf" might become "the good wolf")

            If this character MATCHES an existing one, output:
            {{"DUPLICATE_OF": "existing_character_name", "reason": "why they are the same"}}

            Otherwise, proceed with full character creation.
            """

        # Generate prompt using extracted prompt function
        char_prompt = get_create_character_prompt(
            char_name=char_name,
            char_role=char_role,
            char_importance=char_importance,
            story_title=story.structure.title,
            story_theme=story.structure.theme,
            difficulty=story.preferences.difficulty,
            existing_chars_context=existing_chars_context,
            arc_description=arc_description,
            skill_requirement=skill_requirement,
            min_skills=min_skills,
            arc_milestones=arc_milestones
        )

        char_task = Task(
            description=char_prompt,
            agent=self.character_agent,
            expected_output=f"Valid JSON with skills_learned array containing {min_skills}+ CharacterSkill objects"
        )

        # Execute (in separate thread to avoid blocking event loop)
        crew = Crew(
            agents=[self.character_agent],
            tasks=[char_task],
            process=Process.sequential,
            verbose=True
        )

        self.logger.info(f"🚀 Executing CharacterAgent for {char_name}...")
        try:
            char_result = await self._run_crew_async(crew)
        except Exception as crew_error:
            self.logger.error("CharacterAgent", f"CrewAI execution failed for {char_name}: {crew_error}")
            raise RuntimeError(f"CrewAI error: {str(crew_error)[:60]}") from crew_error

        # Validate we got a result
        if char_result is None:
            self.logger.error("CharacterAgent", f"CrewAI returned None for {char_name}")
            raise RuntimeError("CrewAI returned empty result")

        result_str_preview = str(char_result)[:200] if char_result else "None"
        self.logger.info(f"📤 CharacterAgent returned result for {char_name} ({len(str(char_result))} chars)")
        self.logger.info(f"   📄 Preview: {result_str_preview}...")

        try:
            # Clean up markdown code blocks if present (Claude wraps JSON in ``` markers)
            result_str = str(char_result).strip()

            # Check for empty result
            if not result_str or result_str in ['None', 'null', '']:
                self.logger.error("CharacterAgent", f"CharacterAgent returned empty result for {char_name}")
                raise RuntimeError("Empty/null result from CharacterAgent")

            # Extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*\})\s*```', result_str)
            if json_match:
                result_str = json_match.group(1)
            else:
                # Fallback: simple strip if no code block found
                if result_str.startswith("```json"):
                    result_str = result_str[7:]
                if result_str.startswith("```"):
                    result_str = result_str[3:]
                if result_str.endswith("```"):
                    result_str = result_str[:-3]
                result_str = result_str.strip()

            # Clean common JSON issues from LLM output
            result_str = clean_json_for_character(result_str, char_name)

            char_data = json.loads(result_str)
            self.logger.info(f"✅ Successfully parsed character JSON for {char_name}")

            # === CHECK FOR DUPLICATE DETECTION ===
            # CharacterAgent may have detected this is a duplicate of an existing character
            if "DUPLICATE_OF" in char_data:
                duplicate_of = char_data.get("DUPLICATE_OF", "unknown")
                reason = char_data.get("reason", "No reason given")
                self.logger.info(f"🔄 CharacterAgent detected duplicate: {char_name} = {duplicate_of}")
                self.logger.info(f"   📝 Reason: {reason}")
                # Return None to signal this character was skipped (not an error)
                return None

            # Validate required progression fields
            progression = char_data.get("progression", {})
            skills = progression.get("skills_learned", [])
            emotional_state = progression.get("current_emotional_state", "neutral")

            # Warn if skills are missing or insufficient
            if not skills:
                self.logger.warning(f"⚠️ Character {char_name} has NO skills - adding defaults based on role")
                # Add default skills based on character role
                default_skills = [
                    {"name": "Awareness", "level": 1, "acquired_chapter": 0, "description": f"Basic alertness typical for a {char_role}"}
                ]
                if char_importance in ["major", "supporting"]:
                    default_skills.append({"name": "Communication", "level": 1, "acquired_chapter": 0, "description": "Ability to express thoughts clearly"})
                if char_importance == "major":
                    default_skills.append({"name": "Determination", "level": 2, "acquired_chapter": 0, "description": "Inner drive to overcome obstacles"})
                if "progression" not in char_data:
                    char_data["progression"] = {}
                char_data["progression"]["skills_learned"] = default_skills
            elif len(skills) < min_skills:
                self.logger.warning(f"⚠️ Character {char_name} has only {len(skills)} skills (expected {min_skills}+)")

            # Warn if emotional state is generic "neutral" or missing
            if not emotional_state or (isinstance(emotional_state, str) and emotional_state.lower() == "neutral"):
                self.logger.warning(f"⚠️ Character {char_name} has '{emotional_state}' emotional state - should be specific (eager, anxious, etc.)")
                # Ensure progression dict exists before setting emotional state
                if "progression" not in char_data:
                    char_data["progression"] = {}
                char_data["progression"]["current_emotional_state"] = "curious"  # Default to something more specific

            # Normalize character_arc to Dict[str, str] format
            if 'character_arc' in char_data and char_data['character_arc'] is not None:
                arc = char_data['character_arc']

                if isinstance(arc, str):
                    # LLM returned a string like "becomes brave" or "From impulsive to wise"
                    # Convert to dict with chapter 1 as default
                    arc_text = arc.strip()
                    if arc_text:
                        # Try to extract chapter number from string like "Chapter 3: becomes wise"
                        match = re.match(r'(?:Chapter\s*)?(\d+)[:\s]*(.+)', arc_text, re.IGNORECASE)
                        if match:
                            char_data['character_arc'] = {match.group(1): match.group(2).strip()}
                        else:
                            # No chapter specified, assume it's the overall arc description for chapter 1
                            char_data['character_arc'] = {'1': arc_text}
                    else:
                        char_data['character_arc'] = None

                elif isinstance(arc, list):
                    if arc:
                        char_data['character_arc'] = {str(i+1): str(arc[i]) for i in range(len(arc)) if arc[i] is not None}
                        if not char_data['character_arc']:
                            char_data['character_arc'] = None
                    else:
                        char_data['character_arc'] = None

                elif isinstance(arc, dict):
                    normalized_arc = {}
                    for key, value in arc.items():
                        if value is None:
                            continue
                        elif isinstance(value, list):
                            normalized_arc[str(key)] = "; ".join(str(v) for v in value if v is not None)
                        else:
                            normalized_arc[str(key)] = str(value)
                    char_data['character_arc'] = normalized_arc if normalized_arc else None

                else:
                    # Unknown type - convert to string and use as arc description
                    char_data['character_arc'] = {'1': str(arc)}

            # Try to create Character, with auto-fix on validation errors
            try:
                character = Character(**char_data)
            except ValidationError as first_validation_error:
                # Try to auto-fix common validation issues
                self.logger.warning(f"⚠️ Validation failed for {char_name}, attempting auto-fix...")
                char_data = auto_fix_character_data(char_data, char_name, first_validation_error)
                # Try again with fixed data
                character = Character(**char_data)
                self.logger.info(f"   ✅ Auto-fix succeeded for {char_name}")

            # Save character to Firebase
            self.logger.info(f"💾 Saving {char_name} to Firebase...")
            await self.storage.save_character(story_id, character)
            self.logger.info(f"✅ Character {char_name} saved successfully to Firebase")

            # === BILL VALIDATION: Check for duplicates and historical accuracy ===
            # This runs right after creation to flag issues early
            story_context = f"Title: {story.structure.title}, Theme: {story.structure.theme}, Era: Viking Age (c. 850-930 CE)"
            validation_result = await self._validate_character_historical_accuracy(
                story_id, character, story_context,
                existing_characters=existing_character_objects  # For dedupe check
            )

            # Emit validation event for tracking
            await story_events.emit("character_validated", story_id, {
                "character_name": character.name,
                "is_duplicate": validation_result.get('is_duplicate', False),
                "duplicate_of": validation_result.get('duplicate_of'),
                "is_historical": validation_result.get('is_historical', False),
                "timeline_valid": validation_result.get('timeline_valid'),
                "verdict": validation_result.get('verdict', 'unknown'),
                "suggestions": validation_result.get('suggestions', [])
            })

            # Emit character_ready event with data
            event_data = {
                "name": character.name,
                "role": character.role,
                "background": character.background[:200] + "..." if len(character.background) > 200 else character.background
            }
            await story_events.emit("character_ready", story_id, event_data)
            # NOTE: CompanionAgent handles user communication for character_ready

            self.logger.info(f"✅ Character created: {character.name} ({character.role})")
            return character.model_dump()

        except json.JSONDecodeError as e:
            self.logger.error("CharacterAgent", f"Failed to parse JSON for character {char_name}", e)
            raise RuntimeError(f"JSON parse error: {str(e)[:60]}") from e
        except ValidationError as e:
            # Log full validation error details for debugging (auto-fix already failed)
            self.logger.error("CharacterAgent", f"Failed to create character {char_name} (ValidationError after auto-fix)", e)
            self.logger.info(f"   🔍 Validation errors: {e.error_count()} issues")
            error_summary = []
            for error in e.errors():
                loc = " -> ".join(str(l) for l in error['loc'])
                self.logger.info(f"   ❌ Field '{loc}': {error['msg']} (type: {error['type']})")
                error_summary.append(f"{loc}: {error['msg']}")
                if 'input' in error:
                    input_preview = str(error['input'])[:100]
                    self.logger.info(f"      Input received: {input_preview}")
            # Re-raise with summary for event logging
            raise RuntimeError(f"Validation: {'; '.join(error_summary[:2])}") from e
        except Exception as e:
            import traceback
            self.logger.error("CharacterAgent", f"Failed to create character {char_name} ({type(e).__name__})", e)
            self.logger.info(f"   🔍 Full traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"{type(e).__name__}: {str(e)[:60]}") from e

    async def _validate_character_historical_accuracy(
        self,
        story_id: str,
        character: Character,
        story_context: str,
        existing_characters: List[Character] = None
    ) -> Dict[str, Any]:
        """
        Bill Nye validates a character card for historical accuracy AND duplicates.

        Called immediately after character creation to:
        1. Check if this is a DUPLICATE of an existing character (same person, different name)
        2. Identify if character is FICTIONAL or HISTORICAL
        3. If historical, validate timeline accuracy
        4. Flag any anachronisms (e.g., Eirik Bloodaxe can't be Harald's contemporary)

        Args:
            story_id: Story ID
            character: The Character object to validate
            story_context: Brief context about the story setting/era
            existing_characters: List of existing Character objects for dedupe check

        Returns:
            Dict with validation results including is_duplicate, is_historical, timeline_valid, suggestions
        """
        from crewai import Task, Crew, Process

        self.logger.info(f"🔬 Bill validating character: {character.name}")

        # Build character card for review
        character_card = f"""
CHARACTER CARD FOR VALIDATION
=============================
Name: {character.name}
Role: {character.role}
Age: {character.age}
Background: {character.background}
Personality: {', '.join(character.personality_traits)}
Motivation: {character.motivation}
Relationships: {json.dumps(character.relationships) if character.relationships else 'None specified'}
"""

        # Build existing characters summary for dedupe check
        existing_chars_summary = ""
        if existing_characters and len(existing_characters) > 0:
            chars_list = []
            for c in existing_characters:
                if c.id != character.id:  # Don't compare to self
                    chars_list.append(f"- {c.name} ({c.role}): {c.background[:100]}...")
            if chars_list:
                existing_chars_summary = f"""
=== EXISTING CHARACTERS (CHECK FOR DUPLICATES) ===
{chr(10).join(chars_list)}
"""

        # Generate prompt using extracted prompt function
        validation_prompt = get_validate_character_prompt(
            character_name=character.name,
            character_card=character_card,
            story_context=story_context,
            existing_chars_summary=existing_chars_summary
        )

        validation_task = Task(
            description=validation_prompt,
            agent=self.factcheck_agent,
            expected_output="Valid JSON with historical validation results"
        )

        crew = Crew(
            agents=[self.factcheck_agent],
            tasks=[validation_task],
            process=Process.sequential,
            verbose=False  # Keep quiet for this quick check
        )

        try:
            result = await self._run_crew_async(crew)
            result_str = clean_json_output(str(result))
            validation_data = json.loads(result_str)

            # Log the result
            verdict = validation_data.get('verdict', 'unknown')
            is_duplicate = validation_data.get('is_duplicate', False)
            is_historical = validation_data.get('is_historical', False)

            # Check for duplicates first
            if is_duplicate:
                duplicate_of = validation_data.get('duplicate_of', 'unknown')
                duplicate_reason = validation_data.get('duplicate_reason', '')
                self.logger.warning(f"   ⚠️ {character.name}: DUPLICATE of '{duplicate_of}'")
                self.logger.warning(f"      Reason: {duplicate_reason}")
            elif is_historical:
                timeline_valid = validation_data.get('timeline_valid', True)
                if timeline_valid:
                    self.logger.info(f"   ✅ {character.name}: Historical figure, timeline VALID")
                else:
                    self.logger.warning(f"   ⚠️ {character.name}: Historical figure, timeline ISSUE")
                    for suggestion in validation_data.get('suggestions', []):
                        self.logger.warning(f"      💡 {suggestion}")
            else:
                self.logger.info(f"   ✅ {character.name}: Fictional character (no historical check needed)")

            return validation_data

        except Exception as e:
            self.logger.warning(f"   ⚠️ Bill validation failed for {character.name}: {e}")
            return {
                "character_name": character.name,
                "is_duplicate": None,
                "duplicate_of": None,
                "is_historical": None,
                "timeline_valid": None,
                "error": str(e),
                "verdict": "skipped"
            }

    async def evolve_characters_post_chapter(
        self,
        story_id: str,
        chapter_number: int,
        chapter_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update character progression after a chapter is written.

        This is called automatically after write_chapter() succeeds to evolve
        characters based on what happened in the chapter (D&D-style leveling).

        Args:
            story_id: Story ID
            chapter_number: Chapter that was just completed
            chapter_content: The chapter data including content and characters_featured

        Returns:
            Dict with evolution results
        """
        story = await self.storage.get_story(story_id)
        characters = await self.storage.get_characters(story_id)

        # Get characters featured in this chapter - use identity matching for variant names
        featured_char_names = chapter_content.get('characters_featured', [])
        featured_chars = []
        name_evolutions = []  # Track name changes for logging

        for featured_name in featured_char_names:
            # First try exact match
            exact_match = next((c for c in characters if c.name == featured_name), None)
            if exact_match:
                if exact_match not in featured_chars:
                    featured_chars.append(exact_match)
            else:
                # Use identity matching for variant names
                matched_char = match_character_identity(featured_name, characters)
                if matched_char and matched_char not in featured_chars:
                    # Check if this is a name evolution (new epithet earned)
                    if featured_name != matched_char.name:
                        # Track the old name in name_history
                        if not hasattr(matched_char, 'name_history') or matched_char.name_history is None:
                            matched_char.name_history = []
                        if matched_char.name not in matched_char.name_history:
                            matched_char.name_history.append(matched_char.name)
                        old_name = matched_char.name
                        matched_char.name = featured_name
                        name_evolutions.append((old_name, featured_name))
                        self.logger.info(f"📝 Character evolved: '{old_name}' → '{featured_name}'")
                    featured_chars.append(matched_char)

        if not featured_chars:
            return {"success": True, "characters_evolved": 0, "message": "No characters featured"}

        print(f"   🎭 Evolving {len(featured_chars)} characters based on chapter events...")
        if name_evolutions:
            for old_name, new_name in name_evolutions:
                print(f"   📝 Name evolution: '{old_name}' → '{new_name}'")

        characters_evolved = []

        for character in featured_chars:
            # Check if this chapter has a milestone for this character
            chapter_outline = None
            if story.structure:
                for ch in story.structure.chapters:
                    if ch.number == chapter_number:
                        chapter_outline = ch
                        break

            planned_milestone = None
            if chapter_outline and chapter_outline.character_development_milestones:
                planned_milestone = chapter_outline.character_development_milestones.get(character.name)

            arc_milestone = None
            if character.character_arc:
                arc_milestone = character.character_arc.get(str(chapter_number))

            # Create evolution task
            # Prepare character state summary
            skills_summary = [{'name': s.name, 'level': s.level} for s in character.progression.skills_learned]

            evolution_task = Task(
                description=f"""Analyze {character.name}'s character development in Chapter {chapter_number}.

                CURRENT CHARACTER STATE:
                - Skills: {json.dumps(skills_summary)}
                - Personality traits: {character.personality_traits}
                - Emotional state: {character.progression.current_emotional_state}
                - Existing relationships: {list(character.relationships.keys())}

                WHAT HAPPENED IN CHAPTER {chapter_number}:
                Title: {chapter_content.get('title')}
                Key events: {chapter_content['content'][:800]}...

                PLANNED DEVELOPMENT FOR THIS CHAPTER:
                {planned_milestone or "No specific milestone planned"}

                CHARACTER ARC MILESTONE (if planned):
                {arc_milestone or "No arc milestone for this chapter"}

                Based on the chapter events and planned development, determine if {character.name}:

                1. SKILLS: Learned a NEW skill or IMPROVED an existing one?
                   - Only add skills if they demonstrated them in the chapter
                   - Improve level by 1-2 if they practiced/excelled at existing skill
                   - Common skills: Combat, Strategy, Diplomacy, Leadership, Tracking, Magic, etc.

                2. PERSONALITY: Did a personality trait evolve?
                   - Example: "impulsive" → "thoughtful" after learning a hard lesson
                   - Only if chapter shows clear character growth
                   - Explain what triggered the change

                3. RELATIONSHIPS: New or changed relationship with another character?
                   - Did they bond with someone? Strength increases
                   - Did they have conflict? Relationship changes
                   - New character introduced? New relationship formed

                4. EMOTIONAL STATE: What's their current emotional state now?
                   - Examples: determined, anxious, confident, burdened, hopeful, etc.

                Output ONLY valid JSON with these fields:
                - skills_gained: array of new skills (name, level, acquired_chapter, description)
                - skills_improved: array of improved skills (name, old_level, new_level)
                - personality_changes: array of trait evolution (chapter_number, from_trait, to_trait, trigger_event)
                - relationship_changes: array of relationship updates (chapter_number, other_character, relationship_type, strength, description)
                - new_emotional_state: current emotional state string

                Return empty arrays for any category where NO changes occurred.
                Be conservative - only record genuine development shown in the chapter.""",
                agent=self.character_agent,
                expected_output="Valid JSON with character progression updates"
            )

            crew = Crew(
                agents=[self.character_agent],
                tasks=[evolution_task],
                process=Process.sequential,
                verbose=True
            )

            evolution_result = await self._run_crew_async(crew)

            try:
                from src.models import CharacterSkill, PersonalityEvolution, RelationshipChange

                evolution_str = clean_json_output(str(evolution_result))
                evolution_data = json.loads(evolution_str)

                # Apply skill gains
                for skill_data in evolution_data.get('skills_gained', []):
                    new_skill = CharacterSkill(**skill_data)
                    character.progression.skills_learned.append(new_skill)
                    print(f"      {character.name} learned: {new_skill.name} (Level {new_skill.level})")

                # Apply skill improvements
                for skill_improvement in evolution_data.get('skills_improved', []):
                    skill_name = skill_improvement['name']
                    new_level = skill_improvement['new_level']
                    for skill in character.progression.skills_learned:
                        if skill.name == skill_name:
                            old_level = skill.level
                            skill.level = new_level
                            print(f"      {character.name} improved {skill_name}: Level {old_level} → {new_level}")
                            break

                # Apply personality evolution
                for personality_change in evolution_data.get('personality_changes', []):
                    evo = PersonalityEvolution(**personality_change)
                    character.progression.personality_evolution.append(evo)

                    # Update current traits list
                    if evo.from_trait in character.personality_traits:
                        idx = character.personality_traits.index(evo.from_trait)
                        character.personality_traits[idx] = evo.to_trait
                        print(f"      {character.name}: {evo.from_trait} → {evo.to_trait}")

                # Apply relationship changes
                for rel_change in evolution_data.get('relationship_changes', []):
                    rel = RelationshipChange(**rel_change)
                    character.progression.relationship_changes.append(rel)

                    # Update relationships dict
                    character.relationships[rel.other_character] = f"{rel.relationship_type} (strength: {rel.strength})"
                    print(f"      {character.name} <-> {rel.other_character}: {rel.relationship_type} ({rel.strength}/10)")

                # Update emotional state
                if 'new_emotional_state' in evolution_data:
                    character.progression.current_emotional_state = evolution_data['new_emotional_state']

                # Track chapter appearance
                if chapter_number not in character.progression.chapters_featured:
                    character.progression.chapters_featured.append(chapter_number)

                # Save updated character
                await self.storage.update_character(story_id, character)

                characters_evolved.append({
                    "name": character.name,
                    "changes": {
                        "skills": len(evolution_data.get('skills_gained', [])) + len(evolution_data.get('skills_improved', [])),
                        "personality": len(evolution_data.get('personality_changes', [])),
                        "relationships": len(evolution_data.get('relationship_changes', []))
                    }
                })

            except (json.JSONDecodeError, Exception) as e:
                print(f"      ⚠️  Warning: Failed to evolve {character.name}: {e}")
                continue

        return {
            "success": True,
            "characters_evolved": len(characters_evolved),
            "details": characters_evolved
        }

    def _determine_fact_check_strictness(self, story: Story) -> str:
        """
        Determine how strict fact-checking should be based on story genre.

        Args:
            story: Story object with preferences

        Returns:
            'strict', 'moderate', or 'relaxed'
        """
        # Strict fact-checking for historical/scientific educational content
        if story.preferences.educational_focus in ["history", "science"]:
            return "strict"

        # Relaxed fact-checking for fantasy/sci-fi (allow magic, but keep internal consistency)
        fantasy_themes = ["fantasy", "magic", "sci-fi", "science fiction", "supernatural"]
        if any(theme.lower() in fantasy_themes for theme in story.preferences.themes):
            return "relaxed"

        # Moderate for everything else
        return "moderate"

    def _get_sensory_requirements(self, target_age: int) -> str:
        """
        Get age-appropriate sensory writing guidance.

        Args:
            target_age: The child's age

        Returns:
            String with sensory writing guidelines appropriate for the age
        """
        if target_age <= 6:
            return """        AGE-SPECIFIC SENSORY DEPTH (Ages 4-6):
        - Simple, concrete sensory details (one per paragraph)
        - Example: "The fire crackled warm and orange."
        - Focus on primary senses: sight, sound, touch
        - Keep descriptions short and vivid"""
        elif target_age <= 9:
            return """        AGE-SPECIFIC SENSORY DEPTH (Ages 7-9):
        - Multiple senses per paragraph, emotional POV begins
        - Example: "Harald's heart beat fast as he watched the flames dance,
          their orange light flickering across his father's stern face."
        - Connect sensory details to character emotions
        - Begin showing internal experience"""
        else:
            return """        AGE-SPECIFIC SENSORY DEPTH (Ages 10+):
        - Layered sensory experience with rich internal life
        - Example: "The acrid smoke caught in Harald's throat, and through
          watering eyes he watched the flames consume everything he had ever
          known. His chest ached with a grief too large to name."
        - Full emotional complexity through physical sensation
        - Use sensory details to reveal character psychology"""

    def _get_agent_model_name(self, agent_type: str) -> str:
        """
        Get the model name used by a specific agent type.

        Args:
            agent_type: Type of agent (e.g., 'narrative', 'character', 'structure')

        Returns:
            Model name string (e.g., 'gpt-5-chat-2025-08-07')
        """
        return self.router.get_model_for_agent(agent_type)

    def _get_routing_mode(self) -> str:
        """
        Get the Azure AI Foundry routing mode currently configured.

        Returns:
            Routing mode string ('quality', 'speed', 'cost')
        """
        from src.config import get_settings
        settings = get_settings()
        return getattr(settings, 'foundry_routing_mode', None) or "quality"

    async def _update_plot_elements_from_continuity_review(
        self,
        story_id: str,
        chapter_number: int,
        reviews: List[Dict[str, Any]]
    ) -> None:
        """
        Update story's plot_elements based on ContinuityAgent's review.

        Extracts new elements introduced, marks resolved elements, and logs at-risk threads.

        Args:
            story_id: Story ID
            chapter_number: Current chapter being reviewed
            reviews: List of all Round Table reviews
        """
        # Find the Continuity review
        continuity_review = None
        for review in reviews:
            if review.get("agent") == "Continuity":
                continuity_review = review
                break

        if not continuity_review:
            self.logger.debug("PlotElements", "No Continuity review found - skipping plot element update")
            return

        # Get fresh story data
        story = await self.storage.get_story(story_id)
        if not story or not story.structure:
            return

        # Import PlotElement model
        from src.models.models import PlotElement, PlotElementType

        # Ensure plot_elements list exists
        if not hasattr(story.structure, 'plot_elements') or story.structure.plot_elements is None:
            story.structure.plot_elements = []

        # Process NEW plot elements
        new_elements = continuity_review.get("plot_elements_new", [])
        for elem_data in new_elements:
            try:
                # Map type string to PlotElementType
                elem_type_str = elem_data.get("type", "mystery").lower()
                # Handle both PlotElementType enum and string
                try:
                    elem_type = PlotElementType(elem_type_str)
                except ValueError:
                    elem_type = PlotElementType.MYSTERY  # Default

                new_elem = PlotElement(
                    name=elem_data.get("name", "Unknown element"),
                    element_type=elem_type,
                    introduced_chapter=chapter_number,
                    setup_text=elem_data.get("setup_text", "")
                )
                story.structure.plot_elements.append(new_elem)
                self.logger.info(f"📌 New plot element: '{new_elem.name}' ({elem_type.value}) introduced in Ch{chapter_number}")
            except Exception as e:
                self.logger.warning(f"⚠️  Could not create PlotElement: {e}")

        # Process RESOLVED plot elements
        resolved_elements = continuity_review.get("plot_elements_resolved", [])
        for resolved in resolved_elements:
            resolved_name = resolved.get("name", "").lower()
            resolution_text = resolved.get("resolution_text", "")

            for elem in story.structure.plot_elements:
                if elem.name.lower() == resolved_name and elem.status == "pending":
                    elem.status = "resolved"
                    elem.resolution_chapter = chapter_number
                    elem.resolution_text = resolution_text
                    self.logger.info(f"✅ Plot element resolved: '{elem.name}' (Ch{elem.introduced_chapter} → Ch{chapter_number})")
                    break

        # Log AT-RISK elements (just for visibility, no data change)
        at_risk = continuity_review.get("plot_elements_at_risk", [])
        for risk in at_risk:
            self.logger.warning(f"⚠️  At-risk plot element: '{risk.get('name')}' - {risk.get('risk', 'needs attention')}")

        # Save updated story structure
        try:
            await self.storage.save_story(story)
            self.logger.info(f"💾 Updated plot_elements: {len(story.structure.plot_elements)} total, {len(new_elements)} new, {len(resolved_elements)} resolved")
        except Exception as e:
            self.logger.error("PlotElements", f"Failed to save updated plot elements: {e}")

    def _extract_statements_from_narrative(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract factual statement tags from narrative content.

        Parses <fact id="stmt_X" p="Y">claim text</fact> tags and creates Statement objects.

        Args:
            content: Narrative text with embedded <fact> tags

        Returns:
            List of statement dicts with {id, text, paragraph_number, sentence_number}
        """
        import re
        from src.models import Statement

        # Pattern to match: <fact id="stmt_1" p="1">claim text</fact>
        pattern = r'<fact\s+id="([^"]+)"\s+p="(\d+)">([^<]+)</fact>'
        matches = re.findall(pattern, content)

        statements = []
        for stmt_id, para_num, text in matches:
            statements.append({
                "id": stmt_id,
                "text": text.strip(),
                "paragraph_number": int(para_num),
                "sentence_number": 0  # Will enhance later if needed
            })

        if statements:
            print(f"   📊 Extracted {len(statements)} factual statements for verification")

        return statements

    # =========================================================================
    # ROUND TABLE REVIEW SYSTEM (PARALLEL EXECUTION)
    # =========================================================================
    # After Nnedi writes a draft, all agents gather around the table to review:
    # - Guillermo (Structure): Pacing, themes, visual coherence
    # - Bill (Facts): Historical/scientific accuracy
    # - Clarissa (Characters): Psychology, arc consistency
    # - Benjamin (Prose): Sentence rhythm, show-don't-tell, read-aloud appeal
    #
    # All 4 agents review SIMULTANEOUSLY using ThreadPoolExecutor
    # If ANY agent blocks → Discussion phase → Revision (max 3 rounds)
    # =========================================================================

    def _run_single_agent_review(
        self,
        agent,
        agent_name: str,
        domain: str,
        chapter_content: str,
        chapter_number: int,
        story,
        characters: List[Character],
        chapter_outline: ChapterOutline,
        loop=None,
        story_id: str = None
    ) -> Dict[str, Any]:
        """
        Run a single agent's review task and return parsed JSON result.

        This method runs one agent in isolation for parallel execution.
        Each agent runs in its own thread via ThreadPoolExecutor.

        Args:
            loop: Event loop for thread-safe event emission
            story_id: Story ID for event emission
        """
        # === EMIT: Agent started (before LLM call) ===
        # Uses fire-and-forget pattern to avoid blocking the worker thread
        if story_id:
            story_events.emit_from_thread(
                EVENT_AGENT_STARTED,
                story_id,
                {
                    "agent": agent_name,
                    "task": f"Reviewing Ch{chapter_number} {domain}",
                    "started_at": datetime.now().isoformat()
                }
            )
        # Build shared review context for all agents
        ctx = build_review_context(story, chapter_number, chapter_content, chapter_outline)

        # === EXISTING CHARACTERS CONTEXT (for all reviewers) ===
        # This helps reviewers understand the character roster without creating new ones
        existing_chars_context = ""
        if characters:
            existing_chars_list = "\n".join([
                f"   - {c.name}: {c.role}" + (f" ({', '.join(c.personality_traits[:2])})" if hasattr(c, 'personality_traits') and c.personality_traits else "")
                for c in characters
            ])
            existing_chars_context = f"""
            === EXISTING CHARACTERS (Reference Only) ===
            These characters exist in the story. Do NOT suggest creating new characters.
            You MAY note character evolution (e.g., name/title changes).

{existing_chars_list}
"""

        # Build task description based on agent type
        if agent_name == "Guillermo":
            task_description = get_guillermo_structure_prompt(
                chapter_number=chapter_number,
                total_chapters=ctx['total_chapters'],
                story_title=ctx['story_title'],
                story_theme=ctx['story_theme'],
                target_age=ctx['target_age'],
                chapter_title=chapter_outline.title,
                chapter_synopsis=chapter_outline.synopsis,
                chapter_content=chapter_content,
                existing_chars_context=existing_chars_context,
                all_chapter_outlines=ctx['all_chapter_outlines'],
                previous_chapter_content=ctx['previous_chapter_content']
            )

        elif agent_name == "Bill":
            facts_to_verify = ctx["current_chapter_outline"].get("facts_to_verify", [])
            task_description = get_bill_facts_prompt(
                chapter_number=chapter_number,
                total_chapters=ctx['total_chapters'],
                story_title=ctx['story_title'],
                target_age=ctx['target_age'],
                themes=story.preferences.themes if story.preferences and story.preferences.themes else [],
                educational_focus=story.preferences.educational_focus if story.preferences else 'general',
                educational_goals=ctx['educational_goals'],
                facts_to_verify=facts_to_verify,
                chapter_content=chapter_content,
                existing_chars_context=existing_chars_context
            )

        elif agent_name == "Clarissa":
            character_names = [c.name for c in characters]

            # Build FULL character profiles with arc milestones for this chapter
            char_milestones = ctx["current_chapter_outline"].get("character_development_milestones", {})
            full_character_profiles = []
            for c in characters:
                profile = {
                    'name': c.name,
                    'role': c.role,
                    'age': c.age,
                    'background': c.background,
                    'personality_traits': c.personality_traits,
                    'motivation': c.motivation,
                    'appearance': c.appearance,
                    'relationships': c.relationships,
                    'current_emotional_state': c.progression.current_emotional_state if c.progression else 'neutral',
                    'arc_milestones': c.character_arc,
                    'expected_milestone_this_chapter': char_milestones.get(c.name, 'No specific milestone')
                }
                full_character_profiles.append(profile)

            task_description = get_clarissa_characters_prompt(
                chapter_number=chapter_number,
                total_chapters=ctx['total_chapters'],
                story_title=ctx['story_title'],
                story_theme=ctx['story_theme'],
                target_age=ctx['target_age'],
                character_names=character_names,
                full_character_profiles=full_character_profiles,
                chapter_content=chapter_content,
                existing_chars_context=existing_chars_context,
                previous_chapter_content=ctx['previous_chapter_content']
            )

        elif agent_name == "Benjamin":
            task_description = get_benjamin_prose_prompt(
                chapter_number=chapter_number,
                total_chapters=ctx['total_chapters'],
                story_title=ctx['story_title'],
                story_theme=ctx['story_theme'],
                target_age=ctx['target_age'],
                chapter_content=chapter_content,
                existing_chars_context=existing_chars_context,
                previous_chapter_content=ctx['previous_chapter_content']
            )

        elif agent_name == "Stephen":
            task_description = get_stephen_tension_prompt(
                chapter_number=chapter_number,
                total_chapters=ctx['total_chapters'],
                story_title=ctx['story_title'],
                story_theme=ctx['story_theme'],
                chapter_content=chapter_content,
                existing_chars_context=existing_chars_context,
                previous_chapter_content=ctx['previous_chapter_content']
            )

        elif agent_name == "Continuity":
            # Get existing plot elements from story structure with FULL details
            existing_plot_elements = []
            if story.structure and hasattr(story.structure, 'plot_elements'):
                existing_plot_elements = story.structure.plot_elements or []

            # Build FULL plot element data with setup_text, resolution info, importance
            full_plot_elements = []
            for pe in existing_plot_elements:
                element = {
                    'name': pe.name if hasattr(pe, 'name') else pe.get('name', 'Unknown'),
                    'type': pe.element_type if hasattr(pe, 'element_type') else pe.get('element_type', 'unknown'),
                    'introduced_chapter': pe.introduced_chapter if hasattr(pe, 'introduced_chapter') else pe.get('introduced_chapter', 1),
                    'setup_text': pe.setup_text if hasattr(pe, 'setup_text') else pe.get('setup_text', ''),
                    'resolution_chapter': pe.resolution_chapter if hasattr(pe, 'resolution_chapter') else pe.get('resolution_chapter'),
                    'resolution_text': pe.resolution_text if hasattr(pe, 'resolution_text') else pe.get('resolution_text'),
                    'status': pe.status if hasattr(pe, 'status') else pe.get('status', 'pending'),
                    'importance': pe.importance if hasattr(pe, 'importance') else pe.get('importance', 'major')
                }
                full_plot_elements.append(element)

            task_description = get_continuity_threads_prompt(
                chapter_number=chapter_number,
                total_chapters=ctx['total_chapters'],
                story_title=ctx['story_title'],
                story_theme=ctx['story_theme'],
                chapter_content=chapter_content,
                existing_chars_context=existing_chars_context,
                full_plot_elements=full_plot_elements,
                previous_chapter_content=ctx['previous_chapter_content']
            )

        # Create single-agent Crew and execute
        task = Task(
            description=task_description,
            expected_output="JSON review",
            agent=agent
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )

        # Set up model tracking for this reviewer
        tracker = ModelTracker.get_instance()
        # Reverse lookup: display name (e.g., "Guillermo") -> agent type (e.g., "structure")
        agent_type = self.router.get_agent_by_display_name(agent_name) or "unknown"
        configured_model = self.router.get_model_for_agent(agent_type)
        tracker.set_current_agent(f"Reviewer_{agent_name}", configured_model=configured_model)

        # LOG: Starting reviewer execution (like NarrativeAgent does)
        start_time = time.time()
        print(f"   🚀 Executing {agent_name} reviewer for Chapter {chapter_number}...")

        # Run synchronously (CrewAI doesn't have native async)
        result = crew.kickoff()

        # Get actual model used from tracker
        actual_model = tracker.get_last_model(f"Reviewer_{agent_name}")

        # LOG: Completed (like NarrativeAgent does)
        duration = time.time() - start_time
        print(f"   ✅ {agent_name} completed in {duration:.1f}s (model: {actual_model})")

        # Calculate duration in milliseconds
        duration_ms = int(duration * 1000)

        # Parse result
        try:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', str(result))
            if json_match:
                review = json.loads(json_match.group())
                if "agent" in review and "verdict" in review:
                    # Add model and domain to review result
                    review["model"] = actual_model
                    review["domain"] = domain
                    review["duration_ms"] = duration_ms

                    # === EMIT: Agent completed (fire-and-forget) ===
                    if story_id:
                        story_events.emit_from_thread(
                            EVENT_AGENT_COMPLETED,
                            story_id,
                            {
                                "agent": agent_name,
                                "duration_ms": duration_ms,
                                "success": True,
                                "result_summary": f"{review.get('verdict', 'unknown')}",
                                "model": actual_model
                            }
                        )

                    # === EMIT: Reviewer verdict (fire-and-forget) ===
                    if story_id:
                        story_events.emit_from_thread(
                            EVENT_REVIEWER_VERDICT,
                            story_id,
                            {
                                "chapter": chapter_number,
                                "reviewer": agent_name,
                                "verdict": review.get('verdict', 'unknown'),
                                "notes": review.get('concern', '')[:200] if review.get('concern') else None,
                                "model": actual_model,
                                "domain": domain,
                                "duration_ms": duration_ms
                            }
                        )
                    return review
        except Exception as e:
            self.logger.debug(f"JSON parsing failed for {agent_name} review: {e}")

        # Return concern (not approve!) if parsing fails - don't auto-approve broken reviews
        self.logger.warning(f"⚠️ {agent_name}'s review could not be parsed - defaulting to CONCERN")
        fallback_review = {
            "agent": agent_name,
            "domain": domain,
            "verdict": "concern",
            "praise": "",
            "concern": f"Review parsing failed for {agent_name} - manual check recommended",
            "suggestion": "Re-run review or check agent output manually",
            "model": actual_model,
            "duration_ms": duration_ms
        }

        # === EMIT: Agent completed (fire-and-forget) ===
        if story_id:
            story_events.emit_from_thread(
                EVENT_AGENT_COMPLETED,
                story_id,
                {
                    "agent": agent_name,
                    "duration_ms": duration_ms,
                    "success": False,  # Parsing failed
                    "result_summary": "concern (parse error)",
                    "model": actual_model
                }
            )

        # === EMIT: Fallback verdict (fire-and-forget) ===
        if story_id:
            story_events.emit_from_thread(
                EVENT_REVIEWER_VERDICT,
                story_id,
                {
                    "chapter": chapter_number,
                    "reviewer": agent_name,
                    "verdict": "concern",
                    "notes": fallback_review["concern"][:200],
                    "model": actual_model,
                    "domain": domain,
                    "duration_ms": duration_ms
                }
            )
        return fallback_review

    async def round_table_review(
        self,
        story_id: str,
        chapter_number: int,
        chapter_content: str,
        chapter_outline: ChapterOutline,
        characters: List[Character]
    ) -> Dict[str, Any]:
        """
        All agents review chapter and discuss concerns.
        Can collectively block and request revision.

        Args:
            story_id: Story ID
            chapter_number: Chapter being reviewed
            chapter_content: The written chapter text
            chapter_outline: Original chapter outline/blueprint
            characters: Characters featured in the chapter

        Returns:
            Dict with decision (approved/approved_with_notes/revise), reviews, and guidance
        """
        story = await self.storage.get_story(story_id)
        self.logger.info(f"\n🪑 [Round Table] Chapter {chapter_number} review begins (PARALLEL, 2 concurrent)...")

        # === OBSERVATORY: Emit round table started ===
        reviewers = ["Guillermo", "Bill", "Clarissa", "Benjamin", "Continuity", "Stephen"]
        await story_events.emit_round_table_started(story_id, chapter_number, reviewers)
        self.logger.info(f"📡 Emitted round_table_started event for story {story_id}, chapter {chapter_number}")

        # ===== ADAPTIVE PARALLEL EXECUTION =====
        # Uses AdaptiveRateLimiter instead of fixed ThreadPoolExecutor(max_workers=2)
        # Starts at max 10 concurrent, scales down on 429s, scales up on success
        limiter = get_rate_limiter()
        current_max = limiter.get_provider_concurrency("azure")
        self.logger.info(f"   🎭 Reviewers starting (adaptive parallelism, up to {current_max} concurrent)...")
        self.logger.info("   ├── 🎬 Guillermo reviewing structure...")
        self.logger.info("   ├── 🔬 Bill reviewing facts...")
        self.logger.info("   ├── 📚 Clarissa reviewing characters...")
        self.logger.info("   ├── ✍️  Benjamin reviewing prose...")
        self.logger.info("   ├── 🔗 Continuity reviewing plot threads...")
        self.logger.info("   └── ⚡ Stephen reviewing tension/hooks...")

        # Get event loop for thread-safe event emission
        loop = asyncio.get_running_loop()

        # Define reviewer configurations
        reviewer_configs = [
            (self.structure_agent, "Guillermo", "structure"),
            (self.factcheck_agent, "Bill", "facts"),
            (self.character_agent, "Clarissa", "characters"),
            (self.line_editor_agent, "Benjamin", "prose"),
            (self.continuity_agent, "Continuity", "continuity"),
            (self.tension_agent, "Stephen", "tension"),
        ]

        async def run_reviewer_with_rate_limit(agent, agent_name: str, domain: str) -> Dict[str, Any]:
            """Run a single reviewer with adaptive rate limiting and retry."""
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with limiter.acquire("azure"):
                        # Run the synchronous review method in a thread pool
                        result = await asyncio.to_thread(
                            self._run_single_agent_review,
                            agent, agent_name, domain,
                            chapter_content, chapter_number, story, characters, chapter_outline,
                            loop, story_id
                        )
                        return result
                except RateLimitError:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"🔄 {agent_name}: Retry {attempt + 1}/{max_retries} after rate limit")
                        continue
                    else:
                        self.logger.error(f"❌ {agent_name}: Failed after {max_retries} rate limit retries")
                        return {
                            "agent": agent_name,
                            "verdict": "concern",
                            "concern": f"Rate limit exceeded after {max_retries} retries",
                            "model": None,
                            "domain": domain
                        }
                except Exception as e:
                    self.logger.error(f"❌ {agent_name}: Review failed: {e}")
                    return {
                        "agent": agent_name,
                        "verdict": "concern",
                        "concern": str(e)[:200],
                        "model": None,
                        "domain": domain
                    }
            # Should not reach here, but safety fallback
            return {
                "agent": agent_name,
                "verdict": "concern",
                "concern": "Unknown error during review",
                "model": None,
                "domain": domain
            }

        # Run all 6 reviewers in parallel with adaptive rate limiting
        reviewer_tasks = [
            run_reviewer_with_rate_limit(agent, name, domain)
            for agent, name, domain in reviewer_configs
        ]

        # Gather with timeout
        reviews = []
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*reviewer_tasks, return_exceptions=True),
                timeout=300  # 5 minute timeout for all reviewers
            )
            # Process results
            for i, result in enumerate(results):
                agent_name, domain = reviewer_configs[i][1], reviewer_configs[i][2]
                if isinstance(result, Exception):
                    self.logger.error(f"❌ {agent_name}: Exception: {result}")
                    reviews.append({
                        "agent": agent_name,
                        "verdict": "concern",
                        "concern": str(result)[:200],
                        "model": None,
                        "domain": domain
                    })
                else:
                    reviews.append(result)
        except asyncio.TimeoutError:
            self.logger.error("ROUND_TABLE", f"Round Table timeout after 300s")
            # Create fallback reviews for any missing reviewers
            completed_agents = {r.get('agent') for r in reviews}
            for agent, agent_name, domain in reviewer_configs:
                if agent_name not in completed_agents:
                    reviews.append({
                        "agent": agent_name,
                        "verdict": "concern",
                        "concern": "Reviewer did not complete in time",
                        "model": None,
                        "domain": domain
                    })

        self.logger.info("   ✅ All 6 reviewers completed simultaneously!")

        # Auto-convert Benjamin's sensory_score to block if "needs_work"
        for review in reviews:
            if review.get("agent") == "Benjamin":
                sensory_score = review.get("sensory_score", "adequate")
                if sensory_score == "needs_work" and review.get("verdict") != "block":
                    self.logger.warning(f"   🔄 AUTO-BLOCK: Benjamin's sensory_score is '{sensory_score}' - upgrading to BLOCK")
                    review["verdict"] = "block"
                    original_concern = review.get("concern", "")
                    review["concern"] = f"[AUTO-BLOCKED: sensory_score = needs_work] {original_concern}"
                    # Emit UPDATED verdict (original was already emitted as approve/concern)
                    await story_events.emit_reviewer_verdict(
                        story_id, chapter_number,
                        "Benjamin",
                        "block",
                        review["concern"][:200],
                        model=review.get('model'),
                        domain=review.get('domain'),
                        duration_ms=review.get('duration_ms')
                    )

        # Log each review result (verdicts already emitted from _run_single_agent_review)
        for review in reviews:
            verdict_emoji = "✅" if review.get("verdict") == "approve" else "⚠️" if review.get("verdict") == "concern" else "🚫"
            self.logger.info(f"   {verdict_emoji} {review.get('agent', 'Unknown')}: {review.get('verdict', 'unknown').upper()}")
            if review.get("concern"):
                self.logger.info(f"      └─ {review.get('concern')[:100]}...")

        # Update plot elements from Continuity review
        await self._update_plot_elements_from_continuity_review(story_id, chapter_number, reviews)

        # Check for blocks and concerns
        blocks = [r for r in reviews if r.get("verdict") == "block"]
        concerns = [r for r in reviews if r.get("verdict") == "concern"]

        # ===== ROUND 2: If blocks exist, facilitate discussion and request revision =====
        # Revisions are ONLY triggered by blockers. Concerns trigger polish, not revision.
        if blocks:
            self.logger.info(f"\n   🛑 {len(blocks)} BLOCK(S) - Facilitating discussion...")

            discussion = await self._facilitate_discussion(
                chapter_content, reviews, chapter_number
            )

            revision_guidance = compile_revision_guidance(reviews, discussion)

            # === OBSERVATORY: Emit revision decision ===
            await story_events.emit_round_table_decision(
                story_id, chapter_number, "revision_needed", 1,
                f"{len(blocks)} blocks, {len(concerns)} concerns"
            )

            return {
                "decision": "revise",
                "reviews": reviews,
                "discussion": discussion,
                "revision_guidance": revision_guidance
            }

        # If concerns exist (no blocks), approve with notes and queue for polish
        if concerns:
            self.logger.info(f"\n   📋 {len(concerns)} concern(s) noted - Approved with notes (concerns trigger polish, not revision)")

            # Collect ALL suggestions for Polish Pass
            all_suggestions = []
            for review in reviews:
                if review.get("suggestion"):
                    all_suggestions.append({
                        "agent": review.get("agent", "Unknown"),
                        "domain": review.get("domain", "general"),
                        "suggestion": review.get("suggestion")
                    })

            if all_suggestions:
                self.logger.info(f"   📝 Collected {len(all_suggestions)} suggestions for Polish Pass")

            # === OBSERVATORY: Emit approved with notes ===
            await story_events.emit_round_table_decision(
                story_id, chapter_number, "approved", 0,
                f"Approved with {len(concerns)} minor concern(s)"
            )

            return {
                "decision": "approved_with_notes",
                "reviews": reviews,
                "collective_notes": [r.get("concern") for r in concerns if r.get("concern")],
                "suggestions_for_polish": all_suggestions  # NEW: Always pass suggestions
            }

        # All approved - but still collect suggestions for Polish Pass
        self.logger.info(f"\n   ✅ All agents APPROVE!")

        # Collect ALL suggestions from reviewers (even on approval)
        all_suggestions = []
        for review in reviews:
            if review.get("suggestion"):
                all_suggestions.append({
                    "agent": review.get("agent", "Unknown"),
                    "domain": review.get("domain", "general"),
                    "suggestion": review.get("suggestion")
                })

        if all_suggestions:
            self.logger.info(f"   📝 Collected {len(all_suggestions)} suggestions for Polish Pass")

        # === OBSERVATORY: Emit full approval ===
        await story_events.emit_round_table_decision(
            story_id, chapter_number, "approved", 0,
            "All reviewers approved"
        )

        return {
            "decision": "approved",
            "reviews": reviews,
            "collective_notes": [],
            "suggestions_for_polish": all_suggestions  # NEW: Always pass suggestions
        }

    async def _facilitate_discussion(
        self,
        chapter_content: str,
        initial_reviews: List[Dict],
        chapter_number: int
    ) -> str:
        """
        Let Nnedi respond to concerns, facilitating the writers' room discussion.

        Args:
            chapter_content: The chapter being discussed
            initial_reviews: Reviews from all agents
            chapter_number: Chapter number

        Returns:
            Nnedi's response and revision plan as JSON string
        """
        concerns_summary = "\n".join([
            f"- {r.get('agent', 'Unknown')}: {r.get('concern', 'No specific concern')}"
            for r in initial_reviews if r.get("verdict") in ["block", "concern"]
        ])

        suggestions_summary = "\n".join([
            f"- {r.get('agent', 'Unknown')}: {r.get('suggestion', 'No suggestion')}"
            for r in initial_reviews if r.get("suggestion")
        ])

        self.logger.info(f"\n   ✍️ Nnedi responding to concerns...")

        # Generate discussion prompt using extracted prompt function
        discussion_prompt = get_discussion_prompt(
            chapter_number=chapter_number,
            concerns_summary=concerns_summary,
            suggestions_summary=suggestions_summary,
            chapter_content_preview=chapter_content[:2000]
        )

        discussion_task = Task(
            description=discussion_prompt,
            expected_output="JSON discussion response",
            agent=self.narrative_agent
        )

        crew = Crew(
            agents=[self.narrative_agent],
            tasks=[discussion_task],
            process=Process.sequential,
            verbose=True
        )

        result = str(await self._run_crew_async(crew))
        self.logger.info(f"   💬 Discussion complete")
        return result

    def _parse_round_table_reviews(self, results_str: str) -> List[Dict[str, Any]]:
        """
        Parse JSON reviews from the Round Table agents.

        Args:
            results_str: Raw output from crew kickoff

        Returns:
            List of review dictionaries
        """
        reviews = []

        # Try to find all JSON objects in the output
        # Pattern matches {...} blocks
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'

        matches = re.findall(json_pattern, results_str)

        for match in matches:
            try:
                data = json.loads(match)
                # Check if this looks like a review (has agent, domain, verdict)
                if "agent" in data and "verdict" in data:
                    reviews.append(data)
            except json.JSONDecodeError:
                continue

        # Ensure we have all FOUR reviewers, fill with CONCERN (not approve!) if missing
        agents_found = {r.get("agent") for r in reviews}
        for agent_name, domain in [("Guillermo", "structure"), ("Bill", "facts"), ("Clarissa", "characters"), ("Benjamin", "prose")]:
            if agent_name not in agents_found:
                self.logger.warning(f"⚠️  Missing review from {agent_name} - defaulting to CONCERN (not auto-approve)")
                reviews.append({
                    "agent": agent_name,
                    "domain": domain,
                    "verdict": "concern",
                    "praise": "",
                    "concern": f"Review from {agent_name} is missing - may need re-run",
                    "suggestion": "Re-run Round Table review to get proper feedback"
                })

        return reviews

    async def _revise_chapter(
        self,
        story_id: str,
        original_content: str,
        revision_guidance: str,
        chapter_number: int,
        chapter_outline: ChapterOutline
    ) -> str:
        """
        Nnedi revises the chapter based on Round Table guidance.

        Args:
            story_id: Story ID for fetching language preferences
            original_content: Original chapter content
            revision_guidance: Compiled guidance from reviews
            chapter_number: Chapter number
            chapter_outline: Chapter outline for reference

        Returns:
            Revised chapter content
        """
        self.logger.info(f"   ✍️ Nnedi revising chapter based on Round Table feedback...")

        # Get story language for the revision
        story = await self.storage.get_story(story_id)
        story_language = story.preferences.language if story and story.preferences else "en"
        lang_style = get_language_style(story_language)

        self.logger.info(f"   📝 Revision language: {lang_style['name']} ({story_language})")

        revision_prompt = get_revise_chapter_prompt(
            chapter_number=chapter_number,
            chapter_title=chapter_outline.title,
            language_name=lang_style['name'],
            prose_instruction=lang_style.get('prose_instruction', ''),
            original_content=original_content,
            revision_guidance=revision_guidance
        )

        revision_task = Task(
            description=revision_prompt,
            expected_output=f"Revised chapter content in {lang_style['name']}",
            agent=self.narrative_agent
        )

        crew = Crew(
            agents=[self.narrative_agent],
            tasks=[revision_task],
            process=Process.sequential,
            verbose=True
        )

        result = str(await self._run_crew_async(crew))

        # Clean up any markdown or extra formatting
        revised_content = result.strip()
        if revised_content.startswith("```"):
            revised_content = revised_content.split("```")[1] if "```" in revised_content else revised_content
            revised_content = revised_content.strip()

        self.logger.info(f"   ✅ Revision complete ({len(revised_content)} chars)")
        return revised_content

    async def _polish_chapter(
        self,
        story_id: str,
        chapter_content: str,
        suggestions: List[Dict[str, str]],
        chapter_number: int
    ) -> str:
        """
        Apply Round Table suggestions to polish the chapter.

        This ALWAYS runs after approval to apply line edits and refinements
        suggested by reviewers (especially Benjamin the Line Editor).

        Args:
            story_id: Story ID for event emission
            chapter_content: Approved chapter content
            suggestions: List of suggestions from reviewers
            chapter_number: Chapter number

        Returns:
            Polished chapter content
        """
        if not suggestions:
            self.logger.info(f"   📝 No suggestions to apply - chapter is already polished")
            return chapter_content

        self.logger.info(f"\n   ✨ POLISH PASS - Applying {len(suggestions)} reviewer suggestions...")

        # Get story language for the polish pass
        story = await self.storage.get_story(story_id)
        story_language = story.preferences.language if story and story.preferences else "en"
        lang_style = get_language_style(story_language)

        self.logger.info(f"   📝 Language: {lang_style['name']} ({story_language})")

        # === OBSERVATORY: Emit polish_started event ===
        await story_events.emit("polish_started", story_id, {
            "chapter": chapter_number,
            "suggestions_count": len(suggestions),
            "agent": "NarrativeAgent",  # Polish is done by NarrativeAgent
            "model": self.router.get_model_for_agent("narrative"),
            "language": story_language,
            "suggestions": [
                f"{s.get('agent', 'Unknown')}: {s.get('suggestion', '')[:60]}..."
                for s in suggestions[:5]  # Limit to first 5 for brevity
            ]
        })

        # Format suggestions for the agent
        suggestions_text = "\n\n".join([
            f"**{s['agent']} ({s['domain']}):**\n{s['suggestion']}"
            for s in suggestions
        ])

        # Log each suggestion being applied
        for s in suggestions:
            preview = s['suggestion'][:100] + "..." if len(s['suggestion']) > 100 else s['suggestion']
            self.logger.info(f"      └─ {s['agent']}: {preview}")

        polish_prompt = get_polish_chapter_prompt(
            chapter_number=chapter_number,
            language_name=lang_style['name'],
            prose_instruction=lang_style.get('prose_instruction', ''),
            chapter_content=chapter_content,
            suggestions_text=suggestions_text
        )

        polish_task = Task(
            description=polish_prompt,
            expected_output=f"Polished chapter content in {lang_style['name']} with suggestions applied",
            agent=self.narrative_agent
        )

        crew = Crew(
            agents=[self.narrative_agent],
            tasks=[polish_task],
            process=Process.sequential,
            verbose=True
        )

        result = str(await self._run_crew_async(crew))

        # Clean up any markdown or extra formatting
        polished_content = result.strip()
        if polished_content.startswith("```"):
            polished_content = polished_content.split("```")[1] if "```" in polished_content else polished_content
            polished_content = polished_content.strip()

        # Calculate changes
        original_words = len(chapter_content.split())
        polished_words = len(polished_content.split())
        word_diff = polished_words - original_words

        self.logger.info(f"   ✅ Polish complete: {original_words} → {polished_words} words ({'+' if word_diff >= 0 else ''}{word_diff})")

        # === OBSERVATORY: Emit polish_completed event ===
        await story_events.emit("polish_completed", story_id, {
            "chapter": chapter_number,
            "word_count_before": original_words,
            "word_count_after": polished_words,
            "word_diff": word_diff,
            "agent": "NarrativeAgent",
            "model": self.router.get_model_for_agent("narrative")
        })

        return polished_content

    async def _refine_language(
        self,
        story_id: str,
        chapter_content: str,
        chapter_number: int,
        language: str
    ) -> str:
        """
        Apply language-specific refinement using configured language refiner.

        This is a dynamic system that checks:
        1. If a language_refiner_{lang} agent exists in models.yaml
        2. If prompts exist at src/prompts/refine/{lang}.py

        If both exist, runs the refiner. Otherwise, skips gracefully.

        This runs AFTER polish pass, before save. It preserves structure
        and plot but improves word choice, idioms, and natural flow for
        the target language.

        Strategy for small models:
        - Process paragraph by paragraph (chunking)
        - Use targeted prompts with specific patterns to fix
        - Include few-shot examples for better understanding

        Args:
            story_id: For logging and events
            chapter_content: Polished chapter text
            chapter_number: For context and logging
            language: Language code (e.g., 'no' for Norwegian)

        Returns:
            Refined content, or original if refinement unavailable
        """
        from src.config import get_settings
        from src.prompts.refine import is_language_supported, get_refiner_for_language
        settings = get_settings()

        # Check if prompts exist for this language
        if not is_language_supported(language):
            self.logger.debug(f"   📝 No refinement prompts for '{language}' (add src/prompts/refine/{language}.py)")
            return chapter_content

        # Check if a language refiner agent is configured
        refiner_agent = f"language_refiner_{language}"
        try:
            model = self.router.get_model_for_agent(refiner_agent)
        except Exception:
            self.logger.debug(f"   📝 No '{refiner_agent}' agent in models.yaml")
            return chapter_content

        # Check if Ollama is configured (for local models)
        if model.startswith("ollama/") and not settings.ollama_base_url:
            self.logger.info(f"   📝 Language refinement skipped: OLLAMA_BASE_URL not configured")
            return chapter_content

        try:
            import litellm

            # Load the language-specific refiner module
            refiner = get_refiner_for_language(language)
            lang_name = getattr(refiner, 'LANGUAGE_NAME', language.upper())

            self.logger.info(f"\n   🌐 {lang_name.upper()} REFINEMENT - Polishing for natural {lang_name} phrasing...")
            self.logger.info(f"      Model: {model}")
            self.logger.info(f"      Strategy: Paragraph-by-paragraph with targeted patterns")

            # Emit event for observatory
            await story_events.emit("language_refinement_started", story_id, {
                "chapter": chapter_number,
                "language": language,
                "model": model,
                "content_length": len(chapter_content)
            })

            # Split into paragraphs for chunked processing
            paragraphs = chapter_content.split('\n\n')
            refined_paragraphs = []
            changes_made = 0

            self.logger.info(f"      Processing {len(paragraphs)} paragraphs...")

            for i, paragraph in enumerate(paragraphs):
                # Skip very short paragraphs (likely whitespace or single words)
                if len(paragraph.strip()) < 30:
                    refined_paragraphs.append(paragraph)
                    continue

                # Refine this paragraph using the language-specific refiner
                refined_para = await self._refine_paragraph(
                    paragraph=paragraph,
                    model=model,
                    settings=settings,
                    refiner=refiner
                )

                # Check if changes were made
                if refined_para != paragraph:
                    changes_made += 1

                refined_paragraphs.append(refined_para)

            # Reconstruct the chapter
            refined_content = '\n\n'.join(refined_paragraphs)

            # Calculate changes
            original_words = len(chapter_content.split())
            refined_words = len(refined_content.split())
            word_diff = refined_words - original_words

            self.logger.info(f"   ✅ {lang_name} refinement complete:")
            self.logger.info(f"      Words: {original_words} → {refined_words} ({'+' if word_diff >= 0 else ''}{word_diff})")
            self.logger.info(f"      Paragraphs changed: {changes_made}/{len(paragraphs)}")

            # Emit completion event
            await story_events.emit("language_refinement_completed", story_id, {
                "chapter": chapter_number,
                "language": language,
                "model": model,
                "word_count_before": original_words,
                "word_count_after": refined_words,
                "paragraphs_changed": changes_made,
                "paragraphs_total": len(paragraphs)
            })

            return refined_content

        except Exception as e:
            self.logger.warning(f"   ⚠️ Language refinement skipped (Ollama unavailable): {e}")
            # Emit skip event
            await story_events.emit("language_refinement_skipped", story_id, {
                "chapter": chapter_number,
                "language": language,
                "reason": str(e)[:100]
            })
            return chapter_content

    async def _refine_paragraph(
        self,
        paragraph: str,
        model: str,
        settings,
        refiner
    ) -> str:
        """
        Refine a single paragraph using language-specific prompts.

        Uses the dynamically loaded refiner module which provides:
        - get_refinement_prompt(paragraph) -> str
        - validate_response(response, original) -> str

        Args:
            paragraph: Single paragraph to refine
            model: LLM model identifier
            settings: App settings with ollama_base_url
            refiner: Language-specific refiner module from src/prompts/refine/{lang}.py

        Returns:
            Refined paragraph, or original if refinement fails
        """
        import litellm

        # Get the language-specific prompt
        prompt = refiner.get_refinement_prompt(paragraph)

        try:
            # Build kwargs for litellm - include api_base only for ollama models
            call_kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500,  # Smaller limit for single paragraph
                "temperature": 0.3,
                "timeout": 60,  # Shorter timeout per paragraph
                "drop_params": True
            }

            # Only set api_base for Ollama models
            if model.startswith("ollama/") and settings.ollama_base_url:
                call_kwargs["api_base"] = settings.ollama_base_url

            response = await litellm.acompletion(**call_kwargs)
            raw_response = response.choices[0].message.content.strip()

            # Use the refiner's validation function
            refined = refiner.validate_response(raw_response, paragraph)

            # Additional sanity check: length shouldn't change drastically (±50%)
            orig_len = len(paragraph)
            refined_len = len(refined)
            if refined_len < orig_len * 0.5 or refined_len > orig_len * 1.5:
                return paragraph  # Too different, probably went wrong

            return refined

        except Exception as e:
            # Silent fail for individual paragraphs, return original
            return paragraph

    async def write_chapter(
        self,
        story_id: str,
        chapter_number: int,
        additional_instructions: str = ""
    ) -> Dict[str, Any]:
        """
        Write a single chapter with Round Table collaborative review.

        This method implements a 4-stage collaborative process:
        1. NarrativeAgent (Nnedi) writes initial draft
        2. Round Table Review:
           - Guillermo (Structure): Reviews pacing, themes, visual coherence
           - Bill (Facts): Reviews historical/scientific accuracy
           - Clarissa (Characters): Reviews psychology, arc consistency
           - If ANY agent blocks → Discussion → Revision (max 3 rounds)
        3. Save final chapter to Firebase
        4. Evolve characters based on chapter events (D&D-style progression)

        Args:
            story_id: Story ID
            chapter_number: Chapter number to write
            additional_instructions: Optional user requests/preferences to incorporate

        Returns:
            Dict with chapter data and round_table_status
        """
        import time
        chapter_start_time = time.time()

        story = await self.storage.get_story(story_id)
        if not story or not story.structure:
            return {"success": False, "error": "Story not ready"}

        # === OBSERVATORY: Emit chapter pipeline start ===
        total_chapters = len(story.structure.chapters) if story.structure.chapters else 0
        await story_events.emit_pipeline_stage(
            story_id, f"chapter_{chapter_number}", "in_progress",
            f"{chapter_number}/{total_chapters}"
        )
        await story_events.emit_agent_started(
            story_id, "NarrativeAgent",
            f"Writing Chapter {chapter_number}",
            {"chapter_number": chapter_number}
        )

        # Get chapter outline
        chapter_outline = None
        for ch in story.structure.chapters:
            if ch.number == chapter_number:
                chapter_outline = ch
                break

        if not chapter_outline:
            return {"success": False, "error": f"Chapter {chapter_number} not found in outline"}

        # Get characters and validate they exist
        self.logger.info(f"📥 Fetching characters from Firebase for story {story_id}...")
        characters = await self.storage.get_characters(story_id)

        # LOG: What characters were fetched
        self.logger.info(f"📊 Retrieved {len(characters)} characters from Firebase:")
        for char in characters:
            self.logger.debug("NarrativeAgent", f"- {char.name} ({char.role}): {len(char.personality_traits)} traits, {len(char.progression.skills_learned)} skills")
            self.logger.debug("NarrativeAgent", f"  Personality: {', '.join(char.personality_traits[:3])}")
            self.logger.debug("NarrativeAgent", f"  Motivation: {char.motivation[:100] if char.motivation else 'None'}...")
            self.logger.debug("NarrativeAgent", f"  Background: {char.background[:100] if char.background else 'None'}...")

        # PREREQUISITE VALIDATION: Ensure characters were created
        expected_char_count = len(story.structure.characters_needed) if story.structure.characters_needed else 0
        actual_char_count = len(characters)

        if expected_char_count > 0 and actual_char_count == 0:
            error_msg = f"Cannot write chapter: No characters found. Expected {expected_char_count} characters."
            self.logger.error("NarrativeAgent", error_msg)
            return {"success": False, "error": error_msg}

        if actual_char_count < expected_char_count:
            self.logger.warning(
                f"⚠️  Writing chapter with incomplete character roster: "
                f"{actual_char_count}/{expected_char_count} characters available"
            )

        # =========================================================================
        # STAGE 1: NNEDI WRITES INITIAL DRAFT
        # =========================================================================
        self.logger.info(f"📝 Building task description for NarrativeAgent (Chapter {chapter_number})...")

        # Add special Chapter 1 scene-setting requirements
        chapter_1_context = ""
        if chapter_number == 1:
            main_char = characters[0].name if characters else "the protagonist"
            chapter_1_context = f"""
        FIRST CHAPTER REQUIREMENTS (This is Chapter 1 - the reader's entry point):
        - Spend the first 200+ words on SETTING and ATMOSPHERE
        - Ground the reader in the physical world before any plot
        - Show the world through {main_char}'s sensory experience
        - Build atmosphere before introducing the story's central question
        - Do NOT rush to the inciting incident
        - First 2-3 paragraphs: Physical setting, sensory details, atmosphere
        - Then: {main_char} in their normal world
        - Finally: Hint at what will disrupt this world
"""

        # =========================================================================
        # BUILD "STORY SO FAR" CONTEXT FOR CHAPTERS 2+
        # Critical for character evolution! Hans meets Gyda in Ch2, they kiss in Ch4...
        # =========================================================================
        story_so_far_context = ""
        if chapter_number > 1 and story.chapters:
            previous_chapters = sorted(
                [ch for ch in story.chapters if ch.number < chapter_number and ch.content],
                key=lambda x: x.number
            )

            if previous_chapters:
                story_so_far_context = "\n=== STORY SO FAR (CRITICAL - READ CAREFULLY!) ===\n"
                story_so_far_context += "You MUST continue from where the story left off. Characters have EVOLVED!\n\n"

                for prev_ch in previous_chapters:
                    # Include summary or first ~500 chars of each previous chapter
                    content_preview = prev_ch.content[:800] if prev_ch.content else ""
                    if len(prev_ch.content or "") > 800:
                        content_preview += "..."

                    story_so_far_context += f"--- CHAPTER {prev_ch.number}: {prev_ch.title} ---\n"
                    story_so_far_context += f"{content_preview}\n\n"

                # Add character evolution notes from previous chapters
                story_so_far_context += "=== CHARACTER DEVELOPMENTS FROM PREVIOUS CHAPTERS ===\n"
                for char in characters:
                    if char.progression and char.progression.chapters_featured:
                        prev_featured = [c for c in char.progression.chapters_featured if c < chapter_number]
                        if prev_featured:
                            story_so_far_context += f"\n{char.name}:\n"
                            # Include relationship changes
                            if char.relationships:
                                story_so_far_context += f"  - Relationships: {', '.join([f'{k}: {v}' for k, v in char.relationships.items()])}\n"
                            # Include personality evolution
                            if char.progression.personality_evolution:
                                for evo in char.progression.personality_evolution:
                                    story_so_far_context += f"  - Evolved from '{evo.from_trait}' to '{evo.to_trait}' (trigger: {evo.trigger_event})\n"
                            # Include skills learned
                            if char.progression.skills_learned:
                                story_so_far_context += f"  - Skills learned: {', '.join([s.skill_name for s in char.progression.skills_learned])}\n"
                            # Include current emotional state
                            if char.progression.current_emotional_state:
                                story_so_far_context += f"  - Current emotional state: {char.progression.current_emotional_state}\n"

                story_so_far_context += "\n"
                self.logger.info(f"📚 Built story context: {len(previous_chapters)} previous chapters, {len(story_so_far_context)} chars")

        # Get target_age from preferences (set from child profile during init)
        target_age = 10  # Default
        if story.preferences and story.preferences.target_age:
            target_age = story.preferences.target_age
        elif story.preferences:
            # Fallback: derive from difficulty level if target_age not set
            difficulty = getattr(story.preferences, 'difficulty', None)
            difficulty_str = str(difficulty.value if hasattr(difficulty, 'value') else difficulty).lower()
            if difficulty_str == 'easy':
                target_age = 6
            elif difficulty_str == 'hard':
                target_age = 12
        sensory_guidance = self._get_sensory_requirements(target_age)

        # Age-appropriate word/sentence limits
        if target_age <= 6:
            word_limits = "300-500 words, sentences max 8-10 words"
        elif target_age <= 9:
            word_limits = "400-700 words, sentences max 12-15 words"
        else:
            word_limits = "500-1000 words, sentences max 18-20 words"

        # Get language instruction for prose generation
        story_language = story.preferences.language if story.preferences else "en"
        language_instruction = get_prose_instruction(story_language)

        # Build user instructions section if provided
        user_requests_section = ""
        if additional_instructions:
            user_requests_section = f"""
=== USER REQUESTS (MUST INCORPORATE) ===
{additional_instructions}
Work these requests into the chapter narrative naturally.
"""

        # Add user style requests from story preferences (collected during CompanionAgent conversation)
        # These are things like "make it funny", "more exciting", "less scary" that user said before story started
        user_style_requests_section = self._format_user_style_requests(
            story.preferences.user_style_requests if story.preferences else []
        )

        draft_description = get_write_chapter_prompt(
            chapter_number=chapter_number,
            chapter_title=chapter_outline.title,
            chapter_synopsis=chapter_outline.synopsis,
            story_title=story.structure.title,
            character_names=[c.name for c in characters],
            educational_points=chapter_outline.educational_points,
            target_age=target_age,
            word_limits=word_limits,
            language_instruction=language_instruction,
            sensory_guidance=sensory_guidance,
            story_so_far_context=story_so_far_context,
            chapter_1_context=chapter_1_context,
            user_requests_section=user_requests_section,
            user_style_requests_section=user_style_requests_section
        )

        json_schema = get_write_chapter_json_schema(
            chapter_number=chapter_number,
            chapter_title=chapter_outline.title,
            chapter_synopsis=chapter_outline.synopsis
        )

        # LOG: Task description summary
        lang_style = get_language_style(story_language)
        self.logger.info(f"📋 Task description prepared:")
        self.logger.info(f"   - Chapter: {chapter_number} '{chapter_outline.title}'")
        self.logger.info(f"   - Target age: {target_age}, limits: {word_limits}")
        self.logger.info(f"   - Language: {lang_style['name']} ({story_language})")
        self.logger.info(f"   - Characters referenced: {', '.join([c.name for c in characters])}")

        draft_task = Task(
            description=draft_description + json_schema,
            agent=self.narrative_agent,
            expected_output="Valid JSON chapter content"
        )

        draft_crew = Crew(
            agents=[self.narrative_agent],
            tasks=[draft_task],
            process=Process.sequential,
            verbose=True
        )

        # Set up model tracking for NarrativeAgent
        tracker = ModelTracker.get_instance()
        configured_model = self.router.get_model_for_agent("narrative")
        tracker.set_current_agent(f"NarrativeAgent_Ch{chapter_number}", configured_model=configured_model)

        self.logger.info(f"🚀 Executing NarrativeAgent for Chapter {chapter_number}...")
        generation_started_at = datetime.utcnow()
        draft_result = await self._run_crew_async(draft_crew)
        generation_completed_at = datetime.utcnow()
        generation_duration = (generation_completed_at - generation_started_at).total_seconds()

        # Get actual model used from tracker
        narrative_model = tracker.get_last_model(f"NarrativeAgent_Ch{chapter_number}")

        # LOG: Agent output
        self.logger.info(f"✅ NarrativeAgent completed execution in {generation_duration:.1f}s (model: {narrative_model})")
        self.logger.info(f"📤 Raw output length: {len(str(draft_result))} chars")
        self.logger.debug("NarrativeAgent", f"Raw output preview: {str(draft_result)[:500]}...")

        # Parse draft with improved error handling
        raw_output_str = str(draft_result)
        try:
            result_str = clean_json_output(raw_output_str)
            self.logger.debug("NarrativeAgent", f"Cleaned JSON length: {len(result_str)} chars")

            # Check if we got something that looks like JSON
            if not result_str.strip().startswith('{'):
                self.logger.warning(f"⚠️ Cleaned output doesn't start with '{{': {result_str[:100]}...")

            final_chapter_data = json.loads(result_str, strict=False)
            self.logger.info(f"✅ Successfully parsed chapter JSON")
            self.logger.debug("NarrativeAgent", f"Chapter title: {final_chapter_data.get('title', 'MISSING')}")
            self.logger.debug("NarrativeAgent", f"Content length: {len(final_chapter_data.get('content', ''))} chars")
            self.logger.debug("NarrativeAgent", f"Characters featured: {final_chapter_data.get('characters_featured', [])}")
            self.logger.debug("NarrativeAgent", f"Word count: {final_chapter_data.get('word_count', 'MISSING')}")

            # === NORMALIZE CHARACTER NAMES ===
            # Match NarrativeAgent's character names to existing character names
            # This prevents "Harald Halfdansson" when "Harald Fairhair" exists
            raw_featured = final_chapter_data.get('characters_featured', [])
            normalized_featured = normalize_character_names(raw_featured, characters)
            final_chapter_data['characters_featured'] = normalized_featured
            if raw_featured != normalized_featured:
                self.logger.info(f"📝 Character names normalized: {raw_featured} → {normalized_featured}")

            # === ADD SYNOPSIS FROM CHAPTER OUTLINE ===
            # The Chapter model requires synopsis, but NarrativeAgent may not output it
            # Use the synopsis from the chapter outline (which was used to generate the chapter)
            if 'synopsis' not in final_chapter_data and chapter_outline:
                final_chapter_data['synopsis'] = chapter_outline.synopsis
                self.logger.debug("NarrativeAgent", f"Added synopsis from outline: {chapter_outline.synopsis[:100]}...")

        except json.JSONDecodeError as e:
            # Log detailed error info for debugging
            self.logger.error("NarrativeAgent", f"JSON parsing failed: {e}")
            self.logger.error("NarrativeAgent", f"Raw output length: {len(raw_output_str)} chars")
            self.logger.error("NarrativeAgent", f"Raw output preview: {raw_output_str[:300]}...")
            cleaned_preview = result_str[:300] if 'result_str' in dir() and result_str else 'N/A'
            self.logger.error("NarrativeAgent", f"Cleaned output preview: {cleaned_preview}...")

            # Check if the model output prose instead of JSON (common Gemini issue)
            if '"content"' not in raw_output_str and '"number"' not in raw_output_str:
                self.logger.error("NarrativeAgent", "Output appears to be prose, not JSON - model may have ignored format instructions")

            return {
                "success": False,
                "error": f"Failed to parse chapter draft: {e}",
                "raw_output": raw_output_str,
                "hint": "Model may have output prose instead of JSON"
            }
        except Exception as e:
            self.logger.error("NarrativeAgent", f"Unexpected error parsing chapter: {e}")
            return {
                "success": False,
                "error": f"Failed to parse chapter draft: {e}",
                "raw_output": raw_output_str
            }

        # Extract statements from narrative content
        if 'content' in final_chapter_data:
            statements = self._extract_statements_from_narrative(final_chapter_data['content'])
            if statements:
                final_chapter_data['statements'] = statements

        # =========================================================================
        # STAGE 2: ROUND TABLE REVIEW
        # =========================================================================
        # All agents gather to review: Guillermo (structure), Bill (facts), Clarissa (characters)
        # If ANY agent blocks → Discussion → Revision (max 3 rounds)
        # =========================================================================

        MAX_REVISIONS = 3
        revision_round = 0
        chapter_approved = False
        round_table_result = None

        while not chapter_approved and revision_round < MAX_REVISIONS:
            # Run Round Table review
            round_table_result = await self.round_table_review(
                story_id=story_id,
                chapter_number=chapter_number,
                chapter_content=final_chapter_data.get('content', ''),
                chapter_outline=chapter_outline,
                characters=characters
            )

            if round_table_result["decision"] in ["approved", "approved_with_notes"]:
                chapter_approved = True
                self.logger.info(f"✅ Chapter {chapter_number} APPROVED by Round Table")

                # Store notes if any
                if round_table_result.get("collective_notes"):
                    final_chapter_data["round_table_notes"] = round_table_result["collective_notes"]

                # ===== POLISH PASS: Always apply reviewer suggestions =====
                suggestions = round_table_result.get("suggestions_for_polish", [])
                if suggestions:
                    polished_content = await self._polish_chapter(
                        story_id=story_id,
                        chapter_content=final_chapter_data.get('content', ''),
                        suggestions=suggestions,
                        chapter_number=chapter_number
                    )
                    final_chapter_data['content'] = polished_content
                    final_chapter_data['polished'] = True  # Mark as polished
            else:
                # REVISION NEEDED
                revision_round += 1
                self.logger.info(f"🔄 [Round Table] Revision requested (round {revision_round}/{MAX_REVISIONS})")

                # Nnedi revises based on collective guidance
                revised_content = await self._revise_chapter(
                    story_id=story_id,
                    original_content=final_chapter_data.get('content', ''),
                    revision_guidance=round_table_result.get('revision_guidance', ''),
                    chapter_number=chapter_number,
                    chapter_outline=chapter_outline
                )

                # Update content for next review round
                final_chapter_data['content'] = revised_content

        # ===== CHECK: Handle blocked chapter after MAX_REVISIONS =====
        if not chapter_approved:
            self.logger.warning(f"⚠️ Chapter {chapter_number} NOT APPROVED after {MAX_REVISIONS} revisions - proceeding with warning")

            # Identify which agents are still blocking vs. just have concerns
            blockers = []
            concerns = []
            if round_table_result:
                for review in round_table_result.get("reviews", []):
                    agent = review.get("agent")
                    verdict = review.get("verdict")
                    if verdict == "block":
                        blockers.append(agent)
                    elif verdict == "concern":
                        concerns.append(agent)

            # Build informative message for UI
            # Revisions are ONLY triggered by blockers, concerns trigger polish
            status_parts = []
            if blockers:
                status_parts.append(f"Blockers: {', '.join(blockers)}")
            if concerns:
                status_parts.append(f"Concerns: {', '.join(concerns)}")

            status_message = "; ".join(status_parts) if status_parts else "No explicit blockers"

            # Emit max_revisions_exceeded event for UI
            await story_events.emit(
                "max_revisions_exceeded", story_id,
                {
                    "chapter": chapter_number,
                    "revisions": revision_round,
                    "final_decision": round_table_result.get("decision", "revise") if round_table_result else "unknown",
                    "blockers": blockers,
                    "concerns": concerns,
                    "message": f"Chapter proceeded after {MAX_REVISIONS} revision rounds. {status_message}"
                }
            )

            # Mark chapter with warning flag
            final_chapter_data["approved_with_warning"] = True
            final_chapter_data["max_revisions_note"] = f"Chapter proceeded after {MAX_REVISIONS} revision rounds. {status_message}"

            # Store the blocking concerns for reference
            if round_table_result:
                blocking_concerns = [
                    f"{r.get('agent')}: {r.get('concern', 'No reason given')}"
                    for r in round_table_result.get("reviews", [])
                    if r.get("verdict") == "block"
                ]
                final_chapter_data["blocking_concerns"] = blocking_concerns

        # Build Round Table review for storage
        round_table_review_data = None
        if round_table_result:
            try:
                round_table_review_data = RoundTableReview(
                    decision=round_table_result.get("decision", "approved"),
                    reviews=[AgentReview(**r) for r in round_table_result.get("reviews", [])],
                    discussion=round_table_result.get("discussion"),
                    revision_guidance=round_table_result.get("revision_guidance"),
                    collective_notes=round_table_result.get("collective_notes", []),
                    revision_rounds=revision_round
                )
            except Exception as e:
                self.logger.warning(f"⚠️  Could not create RoundTableReview model: {e}")
                round_table_review_data = None

        # === LANGUAGE REFINEMENT (Norwegian) ===
        # Apply language-specific refinement for Norwegian stories
        # Uses Borealis-4B model via Ollama for natural Norwegian phrasing
        story_language = getattr(story, 'language', None)
        if not story_language and story.preferences:
            story_language = getattr(story.preferences, 'language', 'en')
        story_language = story_language or 'en'

        # LANGUAGE REFINEMENT: Run if configured for this language
        # The _refine_language method checks if both:
        #   1. A language_refiner_{lang} agent exists in models.yaml
        #   2. Prompts exist at src/prompts/refine/{lang}.py
        # If either is missing, it returns the original content unchanged.
        original_content = final_chapter_data.get('content', '')
        refined_content = await self._refine_language(
            story_id=story_id,
            chapter_content=original_content,
            chapter_number=chapter_number,
            language=story_language
        )

        # Only mark as refined if content actually changed
        if refined_content != original_content:
            final_chapter_data['pre_refinement_content'] = original_content
            final_chapter_data['content'] = refined_content
            final_chapter_data['language_refined'] = True

        # STAGE 3: Save final chapter with generation metadata
        try:
            # Create generation metadata for analytics
            generation_metadata = GenerationMetadata(
                model_used=self._get_agent_model_name("narrative"),
                model_provider="azure_foundry",
                generation_started_at=generation_started_at,
                generation_completed_at=generation_completed_at,
                duration_seconds=generation_duration,
                routing_mode=self._get_routing_mode(),
                agent_name="NarrativeAgent"
            )
            final_chapter_data['generation_metadata'] = generation_metadata

            # Add Round Table review to chapter data for persistence
            if round_table_review_data:
                final_chapter_data['round_table_review'] = round_table_review_data

            chapter = Chapter(**final_chapter_data)
            await self.storage.save_chapter(story_id, chapter)

            # Update story status (use update_story_status to avoid overwriting chapters array)
            await self.storage.update_story_status(story_id, StoryStatus.GENERATING_CHAPTER.value)

            # Emit chapter_ready event with data
            event_data = {
                "chapter_number": chapter_number,
                "title": chapter.title,
                "word_count": chapter.word_count,
                "reading_time_minutes": chapter.reading_time_minutes,
                "content_preview": chapter.content[:300] + "..." if len(chapter.content) > 300 else chapter.content
            }
            await story_events.emit("chapter_ready", story_id, event_data)
            # NOTE: CompanionAgent handles user communication for chapter_ready

            # Auto-generate ElevenLabs audio tags in background
            # Uses VoiceDirectorAgent to add [emotion] tags for expressive TTS
            asyncio.create_task(self._auto_generate_audio_tags(story_id, chapter_number))

            # STAGE 4: Evolve characters based on chapter events (D&D-style progression)
            print(f"\n   🎭 Evolving characters based on chapter events...")
            evolution_result = await self.evolve_characters_post_chapter(
                story_id,
                chapter_number,
                final_chapter_data
            )

            # Prepare result
            result = {
                "success": True,
                "chapter": chapter.model_dump(),
                "round_table_status": {
                    "approved": chapter_approved,
                    "decision": round_table_result.get("decision", "approved") if round_table_result else "approved",
                    "revision_rounds": revision_round,
                    "reviews": len(round_table_result.get("reviews", [])) if round_table_result else 0
                },
                "character_evolution": evolution_result
            }

            # Include full Round Table review data if available
            if round_table_review_data:
                result["round_table_review"] = round_table_review_data.model_dump()

            # === OBSERVATORY: Emit chapter completion ===
            duration_ms = int((time.time() - chapter_start_time) * 1000)
            await story_events.emit_agent_completed(
                story_id, "NarrativeAgent", duration_ms, True,
                f"Chapter {chapter_number} written and reviewed",
                model=narrative_model  # Actual model (e.g., "gpt-oss-120b")
            )
            await story_events.emit_pipeline_stage(
                story_id, f"chapter_{chapter_number}", "completed",
                f"{chapter_number}/{total_chapters}",
                {"title": chapter.title if chapter else "", "word_count": chapter.word_count if chapter else 0}
            )

            return result

        except (json.JSONDecodeError, Exception) as e:
            # === OBSERVATORY: Emit error ===
            # Try to get model if tracking was set up, otherwise None
            error_model = None
            try:
                error_model = tracker.get_last_model(f"NarrativeAgent_Ch{chapter_number}")
            except:
                pass
            await story_events.emit_agent_completed(story_id, "NarrativeAgent", 0, False, str(e), model=error_model)
            await story_events.emit_pipeline_stage(story_id, f"chapter_{chapter_number}", "error")
            return {
                "success": False,
                "error": f"Failed to create chapter: {e}",
                "raw_output": str(final_chapter_data)
            }

    async def generate_narrator_commentary(
        self,
        event_type: str,
        story_id: str,
        event_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate real-time narrator commentary using DialogueAgent WITH AUDIO.

        This keeps the conversation flowing while other agents work in the background.
        Now fetches FULL objects from Firebase for rich, detailed commentary.

        Args:
            event_type: Type of event (structure_ready, character_ready, chapter_ready)
            story_id: Story ID
            event_data: Event data from the emitted event

        Returns:
            Dict with:
                - message: Commentary text
                - audio: Base64-encoded audio (or None if failed)
        """
        story = await self.storage.get_story(story_id)
        if not story:
            return {
                "message": "Something exciting is happening with your story!",
                "audio": None
            }

        # Get language for commentary
        story_language = story.preferences.language if story.preferences else "en"
        lang_style = get_language_style(story_language)

        print(f"🎭 Generating narrator commentary for {event_type} in {lang_style['name']}")

        # ===== FETCH FULL OBJECTS FOR RICH CONTEXT =====
        prompt = None
        lang_name = lang_style['name']
        style_instruction = lang_style.get('style_instruction', '')

        if event_type == "structure_ready":
            # Fetch full structure with all details
            structure = story.structure
            if structure:
                chapter_details = ""
                if structure.chapters and len(structure.chapters) > 0:
                    chapter_details = "\nChapters:\n"
                    for ch in structure.chapters[:3]:  # First 3 chapters
                        chapter_details += f"- {ch.title}: {ch.synopsis[:100] if ch.synopsis else 'TBD'}\n"

                educational_goals = ""
                if structure.educational_goals and len(structure.educational_goals) > 0:
                    # Extract 'concept' field from EducationalGoal objects
                    educational_goals = f"\nEducational goals: {', '.join([g.concept for g in structure.educational_goals[:3]])}"

                prompt = get_structure_ready_commentary_prompt(
                    title=structure.title,
                    theme=structure.theme,
                    chapter_count=len(structure.chapters) if structure.chapters else 0,
                    chapter_details=chapter_details,
                    educational_goals=educational_goals,
                    lang_name=lang_name,
                    style_instruction=style_instruction
                )

        elif event_type == "character_ready":
            # Fetch full character object (not just truncated preview)
            char_name = event_data.get('name')
            characters = await self.storage.get_characters(story_id)
            character = next((c for c in characters if c.name == char_name), None)

            if character:
                traits = ", ".join(character.personality_traits[:3]) if character.personality_traits else "mysterious"
                motivation = character.motivation if character.motivation else "their own goals"

                prompt = get_character_ready_commentary_prompt(
                    name=character.name,
                    role=character.role,
                    traits=traits,
                    motivation=motivation,
                    background=character.background,
                    lang_name=lang_name,
                    style_instruction=style_instruction
                )
            else:
                # Fallback if character not found
                prompt = get_character_ready_fallback_prompt(
                    name=char_name,
                    role=event_data.get('role', 'unknown'),
                    lang_name=lang_name
                )

        elif event_type == "chapter_ready":
            # Fetch full chapter object
            chapter_num = event_data.get('chapter_number')
            chapters = await self.storage.get_chapters(story_id)
            chapter = next((ch for ch in chapters if ch.number == chapter_num), None)

            if chapter:
                educational = ""
                if chapter.educational_points and len(chapter.educational_points) > 0:
                    educational = f"\nEducational focus: {', '.join(chapter.educational_points[:2])}"

                vocab = ""
                if chapter.vocabulary_words and len(chapter.vocabulary_words) > 0:
                    # Extract 'word' field from VocabularyWord objects
                    vocab = f"\nNew words: {', '.join([v.word for v in chapter.vocabulary_words[:3]])}"

                prompt = get_chapter_ready_commentary_prompt(
                    chapter_number=chapter_num,
                    title=chapter.title,
                    synopsis=chapter.synopsis[:200] if chapter.synopsis else 'An exciting new chapter',
                    educational=educational,
                    vocab=vocab,
                    word_count=str(event_data.get('word_count', 'TBD')),
                    lang_name=lang_name,
                    style_instruction=style_instruction
                )
            else:
                # Fallback if chapter not found
                prompt = get_chapter_ready_fallback_prompt(
                    chapter_number=chapter_num,
                    title=event_data.get('title', 'Untitled'),
                    lang_name=lang_name
                )

        # Fallback prompt
        if not prompt:
            prompt = get_fallback_commentary_prompt(lang_name=lang_name)

        # Use DialogueAgent to generate commentary
        commentary_task = Task(
            description=prompt,
            agent=self.dialogue_agent,
            expected_output="Brief, enthusiastic narrator commentary (2-3 sentences)"
        )

        crew = Crew(
            agents=[self.dialogue_agent],
            tasks=[commentary_task],
            process=Process.sequential,
            verbose=False  # Don't spam console for commentary
        )

        result = await self._run_crew_async(crew)
        commentary_text = str(result).strip()

        print(f"📝 Generated commentary: {commentary_text[:100]}...")

        # ===== GENERATE AUDIO (dialogue = commentary) =====
        from src.services.voice import voice_service
        tts_language = get_tts_language_code(story_language)
        audio_bytes = await voice_service.text_to_speech(
            text=commentary_text,
            speaking_rate=0.9,  # Slightly slower for kids
            language_code=tts_language,
            use_case="dialogue"
        )

        audio_base64 = None
        if audio_bytes:
            audio_base64 = voice_service.encode_audio_base64(audio_bytes)
            print(f"✅ Commentary audio generated ({lang_style['name']}): {len(audio_base64)} chars base64")
        else:
            print(f"⚠️  Commentary audio generation failed")

        return {
            "message": commentary_text,
            "audio": audio_base64
        }

    def _format_user_style_requests(self, style_requests: list) -> str:
        """
        Format user style requests for inclusion in agent task descriptions.

        Returns an empty string if no requests, or a formatted block if present.
        """
        if not style_requests:
            return ""

        requests_list = "\n".join(f"- {req}" for req in style_requests)
        return f"""
=== USER STYLE REQUESTS (Apply to ENTIRE story) ===
The user has specifically requested the following style/tone preferences.
You MUST incorporate these into your work:
{requests_list}
==================================================="""

    async def create_complete_story(
        self,
        prompt: str,
        user_id: str = None,
        target_age: int = None,
        preferences: Dict[str, Any] = None,
        child_id: str = None,
        language: str = None
    ) -> Dict[str, Any]:
        """
        Complete end-to-end story creation (for testing).

        Args:
            prompt: Story request
            user_id: DEPRECATED - Use child_id. Kept for backward compatibility.
            target_age: Target age. If not provided, derived from child profile.
            preferences: Preferences
            child_id: Child profile ID (new family account model)
            language: Language override. If not provided, fetched from parent account.

        Returns:
            Complete story data
        """
        # Step 1: Initialize
        init_result = await self.initialize_story(
            prompt=prompt,
            user_id=user_id,
            target_age=target_age,
            preferences=preferences,
            child_id=child_id,
            language=language
        )
        if not init_result["success"]:
            return init_result

        story_id = init_result["story_id"]

        # Step 2: Generate structure
        structure_result = await self.generate_story_structure(story_id)
        if not structure_result["success"]:
            return structure_result

        # Step 3: Create characters
        characters_result = await self.create_characters(story_id)
        if not characters_result["success"]:
            return characters_result

        # Step 4: Write first chapter
        chapter_result = await self.write_chapter(story_id, 1)
        if not chapter_result["success"]:
            return chapter_result

        # Get final story
        final_story = await self.storage.get_story(story_id)

        return {
            "success": True,
            "story_id": story_id,
            "welcome_message": init_result["welcome_message"],
            "story": final_story.model_dump() if final_story else None,
            "factcheck_status": chapter_result.get("factcheck_status", {}),
            "factcheck_issues": chapter_result.get("factcheck_issues", [])
        }

    # =========================================================================
    # HYBRID CHAPTER GENERATION SYSTEM
    # =========================================================================

    async def generate_chapter_with_inputs(
        self,
        story_id: str,
        chapter_number: int,
        user_inputs: List[QueuedInput] = None
    ) -> Dict[str, Any]:
        """
        Write a chapter incorporating any queued user inputs.

        This extends write_chapter() to integrate user story choices (Tier 2-4).

        Args:
            story_id: Story ID
            chapter_number: Chapter to write
            user_inputs: List of QueuedInput objects to incorporate

        Returns:
            Dict with chapter data and applied inputs
        """
        story = await self.storage.get_story(story_id)
        if not story or not story.structure:
            return {"success": False, "error": "Story not ready"}

        # Get chapter outline
        chapter_outline = None
        for ch in story.structure.chapters:
            if ch.number == chapter_number:
                chapter_outline = ch
                break

        if not chapter_outline:
            return {"success": False, "error": f"Chapter {chapter_number} not found in outline"}

        # Mark chapter as generating
        await self.storage.update_chapter_status(story_id, chapter_number, ChapterStatus.GENERATING)
        await self.storage.set_generating_chapter(story_id, chapter_number)

        # Emit generating event
        await story_events.emit("chapter_generating", story_id, {
            "chapter_number": chapter_number,
            "title": chapter_outline.title
        })

        # Build user input instructions if any
        user_input_instructions = ""
        if user_inputs and len(user_inputs) > 0:
            for inp in user_inputs:
                if inp.story_direction:
                    user_input_instructions += f"- STORY REQUEST: {inp.story_direction}\n"
                elif inp.preference_updates:
                    # Format preference updates nicely
                    pref_str = str(inp.preference_updates)
                    if isinstance(inp.preference_updates, dict):
                        pref_str = inp.preference_updates.get('style_adjustment', inp.preference_updates.get('adjustment', str(inp.preference_updates)))
                    user_input_instructions += f"- STYLE PREFERENCE: {pref_str}\n"
            self.logger.info(f"📝 Incorporating {len(user_inputs)} user inputs into chapter {chapter_number}")
            self.logger.info(f"   Instructions: {user_input_instructions.strip()}")

            # === OBSERVATORY: Emit user_input_applied event for debug visibility ===
            await story_events.emit("user_input_applied", story_id, {
                "chapter": chapter_number,
                "inputs_count": len(user_inputs),
                "inputs": [
                    {
                        "id": inp.id,
                        "raw": (inp.raw_input or "")[:100],
                        "tier": inp.tier.value if hasattr(inp.tier, 'value') else str(inp.tier),
                        "direction": (inp.story_direction or "")[:80] if inp.story_direction else None
                    }
                    for inp in user_inputs
                ]
            })

        # Write the chapter with inputs - PASS THE INSTRUCTIONS!
        result = await self.write_chapter(story_id, chapter_number, additional_instructions=user_input_instructions)

        if result["success"]:
            # Mark inputs as applied
            if user_inputs:
                for inp in user_inputs:
                    await self.storage.mark_input_applied(story_id, inp.id, chapter_number)

            # Update chapter status to READY
            await self.storage.update_chapter_status(story_id, chapter_number, ChapterStatus.READY)

            result["inputs_applied"] = [inp.id for inp in user_inputs] if user_inputs else []

        return result

    async def start_buffered_generation(
        self,
        story_id: str,
        start_chapter: int = 1
    ) -> Dict[str, Any]:
        """
        Start the hybrid buffered chapter generation process.

        This generates Chapter 1 immediately and makes it READY.
        Subsequent chapters are generated when the user starts reading.

        Flow:
        1. Generate Chapter 1 → READY
        2. When user starts reading Chapter 1:
           - Start generating Chapter 2 in background
        3. When user finishes Chapter 1:
           - Chapter 2 should be READY
           - Start generating Chapter 3
        4. Continue pattern until story complete

        Args:
            story_id: Story ID
            start_chapter: Which chapter to start with (default 1)

        Returns:
            Dict with result of first chapter generation
        """
        story = await self.storage.get_story(story_id)
        if not story or not story.structure:
            return {"success": False, "error": "Story structure not ready"}

        total_chapters = len(story.structure.chapters)
        self.logger.info(f"🚀 Starting buffered generation for story {story_id}")
        self.logger.info(f"   Total chapters: {total_chapters}")

        # Initialize reading state
        reading_state = await self.storage.create_reading_state(story_id)

        # Initialize all chapters as PENDING
        for i in range(1, total_chapters + 1):
            reading_state.set_chapter_status(i, ChapterStatus.PENDING)
        await self.storage.update_reading_state(story_id, reading_state)

        # Generate first chapter immediately
        self.logger.info(f"📝 Generating Chapter {start_chapter} (initial buffer)")

        result = await self.generate_chapter_with_inputs(
            story_id,
            start_chapter,
            user_inputs=[]  # No user inputs for first chapter
        )

        if result["success"]:
            # Mark as READY
            await self.storage.update_chapter_status(story_id, start_chapter, ChapterStatus.READY)
            # NOTE: chapter_ready event is emitted by write_chapter() - CompanionAgent handles user communication

            self.logger.info(f"✅ Chapter {start_chapter} ready. Waiting for user to start reading.")

        return result

    async def generate_all_chapters(
        self,
        story_id: str
    ) -> Dict[str, Any]:
        """
        Generate ALL chapters immediately (not buffered).

        Use this for E2E testing and when you want complete stories upfront.
        Unlike start_buffered_generation(), this writes all chapters in sequence
        before returning.

        Args:
            story_id: Story ID

        Returns:
            Dict with success status, chapters_written count, and any errors
        """
        story = await self.storage.get_story(story_id)
        if not story or not story.structure:
            return {"success": False, "error": "Story structure not ready"}

        total_chapters = len(story.structure.chapters)
        self.logger.info(f"🚀 Starting FULL generation for story {story_id}")
        self.logger.info(f"   Total chapters to generate: {total_chapters}")

        # Step 1: Create characters if not already created
        # This is required before writing any chapters
        if not story.characters or len(story.characters) == 0:
            self.logger.info(f"📝 Creating characters first (required for chapter generation)...")
            characters_result = await self.create_characters(story_id)
            if not characters_result["success"]:
                return {
                    "success": False,
                    "error": f"Character creation failed: {characters_result.get('error', 'Unknown error')}",
                    "chapters_written": 0
                }
            self.logger.info(f"✅ Characters created: {characters_result.get('characters_created', 0)}")

            # Refresh story data after character creation
            story = await self.storage.get_story(story_id)

        # Initialize reading state
        reading_state = await self.storage.create_reading_state(story_id)

        # Initialize all chapters as PENDING
        for i in range(1, total_chapters + 1):
            reading_state.set_chapter_status(i, ChapterStatus.PENDING)
        await self.storage.update_reading_state(story_id, reading_state)

        # === V2 REFINEMENT CHECK ===
        # If Chapter 1 was already written (e.g., by story init) but V2 hasn't run,
        # trigger it now before generating remaining chapters
        chapter_1_exists = any(
            ch.number == 1 and ch.content
            for ch in (story.chapters or [])
        )
        v2_already_done = (
            story.structure.refinement_v2 is not None
            if hasattr(story.structure, 'refinement_v2') else False
        )

        if chapter_1_exists and not v2_already_done and total_chapters > 1:
            self.logger.info(f"📚 Structure V2: Chapter 1 exists but V2 not done - triggering refinement...")
            try:
                v2_result = await self.refine_structure_v2(story_id)
                if v2_result.get("success"):
                    self.logger.info(f"✅ Structure V2: Refined {v2_result.get('refined_chapters', 0)} chapter synopses")
                else:
                    self.logger.warning(f"⚠️ Structure V2 returned non-success: {v2_result.get('error', 'unknown')}")
            except Exception as e:
                self.logger.warning(f"⚠️ Structure V2 refinement failed (non-fatal): {e}")

        results = []
        errors = []

        # Retry configuration
        MAX_RETRIES = 3
        RETRY_DELAY_SECONDS = 5

        # Generate ALL chapters in sequence with retry logic
        for chapter_num in range(1, total_chapters + 1):
            self.logger.info(f"📖 Generating chapter {chapter_num}/{total_chapters}")

            # Skip chapters that already exist (e.g., Chapter 1 from story init)
            story = await self.storage.get_story(story_id)
            chapter_already_exists = any(
                ch.number == chapter_num and ch.content
                for ch in (story.chapters or [])
            )
            if chapter_already_exists:
                self.logger.info(f"   ⏭️ Chapter {chapter_num} already exists, skipping...")
                results.append({"success": True, "chapter_number": chapter_num, "skipped": True})
                continue

            # SEQUENTIAL DEPENDENCY CHECK: Can't write Chapter N without Chapter N-1
            # Characters evolve! Plot builds! Hans meets Gyda in Ch2, they kiss in Ch4...
            if chapter_num > 1:
                # Refresh story to check previous chapter
                story = await self.storage.get_story(story_id)
                prev_chapter_exists = any(
                    ch.number == chapter_num - 1 and ch.content
                    for ch in (story.chapters or [])
                )
                if not prev_chapter_exists:
                    error_msg = f"Cannot write Chapter {chapter_num}: Chapter {chapter_num - 1} is missing! Characters evolve across chapters."
                    self.logger.error("ChapterSequence", error_msg)
                    errors.append({"chapter": chapter_num, "error": error_msg})
                    break  # Stop - can't continue without previous chapter

            # Retry loop for chapter generation
            chapter_success = False
            last_error = None

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    self.logger.info(f"   🔄 Attempt {attempt}/{MAX_RETRIES} for Chapter {chapter_num}")

                    result = await self.generate_chapter_with_inputs(
                        story_id,
                        chapter_num,
                        user_inputs=[]  # No queued inputs for bulk generation
                    )

                    if result["success"]:
                        # Mark as READY
                        await self.storage.update_chapter_status(story_id, chapter_num, ChapterStatus.READY)
                        # NOTE: chapter_ready event is emitted by write_chapter() - CompanionAgent handles user communication

                        results.append(result)
                        self.logger.info(f"✅ Chapter {chapter_num}/{total_chapters} complete")
                        chapter_success = True
                        break  # Success! Move to next chapter

                    else:
                        last_error = result.get("error", "Unknown error")
                        self.logger.warning(f"   ⚠️ Attempt {attempt} failed: {last_error}")

                        if attempt < MAX_RETRIES:
                            self.logger.info(f"   ⏳ Waiting {RETRY_DELAY_SECONDS}s before retry...")
                            await asyncio.sleep(RETRY_DELAY_SECONDS)

                except Exception as e:
                    last_error = str(e)
                    self.logger.warning(f"   ⚠️ Attempt {attempt} exception: {last_error}")

                    if attempt < MAX_RETRIES:
                        self.logger.info(f"   ⏳ Waiting {RETRY_DELAY_SECONDS}s before retry...")
                        await asyncio.sleep(RETRY_DELAY_SECONDS)

            # If all retries failed, we must stop - can't skip chapters!
            if not chapter_success:
                error_msg = f"Chapter {chapter_num} failed after {MAX_RETRIES} retries: {last_error}"
                errors.append({"chapter": chapter_num, "error": error_msg, "retries_attempted": MAX_RETRIES})
                self.logger.error("ChapterGeneration", error_msg)

                # CRITICAL: Stop generation - characters evolve across chapters!
                # We cannot write Chapter 3 without Chapter 2's character developments
                self.logger.error("ChapterGeneration", f"🛑 STOPPING: Cannot continue without Chapter {chapter_num}")
                break

            # === STRUCTURE V2: Refine synopses after Chapter 1 (matches WebSocket/Companion flow) ===
            # This is CRITICAL for E2E test fidelity - must trigger same flow as production
            if chapter_num == 1 and chapter_success and total_chapters > 1:
                self.logger.info(f"📚 Structure V2: Refining synopses based on Chapter 1 content...")
                try:
                    v2_result = await self.refine_structure_v2(story_id)
                    if v2_result.get("success"):
                        self.logger.info(f"✅ Structure V2: Refined {v2_result.get('refined_chapters', 0)} chapter synopses")
                        if v2_result.get("user_inputs_incorporated", 0) > 0:
                            self.logger.info(f"   📝 Incorporated {v2_result['user_inputs_incorporated']} user inputs")
                    else:
                        # Non-fatal - original structure is preserved
                        self.logger.warning(f"⚠️  Structure V2: {v2_result.get('error', 'Unknown error')} (using original synopses)")
                except Exception as e:
                    # Non-fatal - log and continue with original structure
                    self.logger.warning(f"⚠️  Structure V2 exception: {str(e)[:100]} (using original synopses)")

        chapters_written = len(results)
        self.logger.info(f"🏁 Full generation complete: {chapters_written}/{total_chapters} chapters written")

        if errors:
            self.logger.warning(f"⚠️  {len(errors)} chapters had errors: {errors}")

        return {
            "success": chapters_written > 0,
            "chapters_written": chapters_written,
            "total_chapters": total_chapters,
            "errors": errors if errors else None
        }

    async def on_user_starts_reading(
        self,
        story_id: str,
        chapter_number: int
    ) -> Dict[str, Any]:
        """
        Called when user starts reading a chapter.

        Triggers buffered generation of the next chapter.

        Args:
            story_id: Story ID
            chapter_number: Chapter user started reading

        Returns:
            Dict with status and next chapter info
        """
        story = await self.storage.get_story(story_id)
        if not story or not story.structure:
            return {"success": False, "error": "Story not ready"}

        total_chapters = len(story.structure.chapters)
        next_chapter = chapter_number + 1

        # Update reading state
        await self.storage.update_reading_position(story_id, chapter_number, 0.0)
        await self.storage.update_chapter_status(story_id, chapter_number, ChapterStatus.READING)

        # Emit reading started event
        await story_events.emit("chapter_reading_started", story_id, {
            "chapter_number": chapter_number
        })

        # Check if there's a next chapter to buffer
        if next_chapter > total_chapters:
            self.logger.info(f"📖 User reading final chapter {chapter_number}")
            return {
                "success": True,
                "reading_chapter": chapter_number,
                "is_final_chapter": True
            }

        # Check if next chapter is already generated
        reading_state = await self.storage.get_reading_state(story_id)
        if reading_state:
            next_status = reading_state.get_chapter_status(next_chapter)
            if next_status in [ChapterStatus.READY, ChapterStatus.GENERATING]:
                self.logger.info(f"📖 Chapter {next_chapter} already {next_status.value}")
                return {
                    "success": True,
                    "reading_chapter": chapter_number,
                    "next_chapter": next_chapter,
                    "next_chapter_status": next_status.value
                }

        # Start generating next chapter in background
        self.logger.info(f"🔄 Starting buffered generation of Chapter {next_chapter}")

        # Get any queued inputs for next chapter
        queued_inputs = await self.storage.get_queued_inputs(story_id, next_chapter)

        # Generate next chapter (this happens while user reads current chapter)
        import asyncio
        asyncio.create_task(self._generate_chapter_background(
            story_id,
            next_chapter,
            queued_inputs
        ))

        return {
            "success": True,
            "reading_chapter": chapter_number,
            "generating_chapter": next_chapter,
            "inputs_queued": len(queued_inputs)
        }

    async def _generate_chapter_background(
        self,
        story_id: str,
        chapter_number: int,
        user_inputs: List[QueuedInput]
    ):
        """
        Background task to generate a chapter.

        Args:
            story_id: Story ID
            chapter_number: Chapter to generate
            user_inputs: Queued inputs to incorporate
        """
        try:
            result = await self.generate_chapter_with_inputs(
                story_id,
                chapter_number,
                user_inputs
            )

            if result["success"]:
                # Mark as READY
                await self.storage.update_chapter_status(
                    story_id, chapter_number, ChapterStatus.READY
                )
                # NOTE: chapter_ready event is emitted by write_chapter() - CompanionAgent handles user communication

                self.logger.info(f"✅ Background generation complete: Chapter {chapter_number}")
            else:
                self.logger.error(
                    "HybridGeneration",
                    f"Background generation failed for chapter {chapter_number}: {result.get('error')}"
                )

        except Exception as e:
            self.logger.error("HybridGeneration", f"Background generation error", e)

    async def on_user_finishes_chapter(
        self,
        story_id: str,
        chapter_number: int
    ) -> Dict[str, Any]:
        """
        Called when user finishes reading a chapter.

        Updates state and checks if next chapter is ready.

        Args:
            story_id: Story ID
            chapter_number: Chapter user finished

        Returns:
            Dict with next chapter status and any bridge content if needed
        """
        story = await self.storage.get_story(story_id)
        if not story or not story.structure:
            return {"success": False, "error": "Story not ready"}

        total_chapters = len(story.structure.chapters)
        next_chapter = chapter_number + 1

        # Mark current chapter as completed
        await self.storage.mark_chapter_completed(story_id, chapter_number)

        # Emit chapter finished event
        await story_events.emit("chapter_reading_finished", story_id, {
            "chapter_number": chapter_number
        })

        # Check if story is complete
        if next_chapter > total_chapters:
            # Story complete!
            await self.storage.update_story_status(story_id, StoryStatus.COMPLETED.value)
            await story_events.emit("story_complete", story_id, {
                "total_chapters": total_chapters
            })

            return {
                "success": True,
                "story_complete": True,
                "total_chapters": total_chapters
            }

        # Check if next chapter is ready
        reading_state = await self.storage.get_reading_state(story_id)
        next_status = reading_state.get_chapter_status(next_chapter) if reading_state else ChapterStatus.PENDING

        if next_status == ChapterStatus.READY:
            # Next chapter is ready
            return {
                "success": True,
                "next_chapter": next_chapter,
                "next_chapter_ready": True
            }
        elif next_status == ChapterStatus.GENERATING:
            # Still generating - provide bridge content
            bridge_content = await self._generate_bridge_content(story_id, chapter_number)
            return {
                "success": True,
                "next_chapter": next_chapter,
                "next_chapter_ready": False,
                "bridge_content": bridge_content
            }
        else:
            # Not started - this shouldn't happen in normal flow
            self.logger.warning(f"⚠️  Next chapter {next_chapter} not started. Starting now.")
            queued_inputs = await self.storage.get_queued_inputs(story_id, next_chapter)
            import asyncio
            asyncio.create_task(self._generate_chapter_background(
                story_id, next_chapter, queued_inputs
            ))

            bridge_content = await self._generate_bridge_content(story_id, chapter_number)
            return {
                "success": True,
                "next_chapter": next_chapter,
                "next_chapter_ready": False,
                "generating_started": True,
                "bridge_content": bridge_content
            }

    async def _generate_bridge_content(
        self,
        story_id: str,
        just_finished_chapter: int
    ) -> Dict[str, Any]:
        """
        Generate engaging bridge content while waiting for next chapter.

        Options:
        - Recap key moments from previous chapter
        - "Did you know?" educational fact
        - Reflection question about the story

        Args:
            story_id: Story ID
            just_finished_chapter: Chapter number just completed

        Returns:
            Dict with bridge content type and message
        """
        story = await self.storage.get_story(story_id)
        chapters = await self.storage.get_chapters(story_id)

        # Get language for bridge content
        story_language = story.preferences.language if story and story.preferences else "en"
        lang_style = get_language_style(story_language)

        # Get the chapter that was just finished
        finished_chapter = None
        for ch in chapters:
            if ch.number == just_finished_chapter:
                finished_chapter = ch
                break

        if not finished_chapter:
            # Fallback messages by language
            fallbacks = {
                "no": "Neste kapittel er nesten klart! Bare et øyeblikk...",
                "es": "¡El próximo capítulo está casi listo! Solo un momento...",
                "en": "The next chapter is almost ready! Just a moment..."
            }
            return {
                "type": "waiting",
                "message": fallbacks.get(story_language, fallbacks["en"])
            }

        # Generate bridge content using DialogueAgent
        bridge_task = Task(
            description=f"""The child just finished reading Chapter {just_finished_chapter}: "{finished_chapter.title}".
            The next chapter is still being prepared.

            LANGUAGE: You MUST respond in {lang_style['name']}.
            {lang_style.get('dialogue_instruction', '')}

            Chapter summary: {finished_chapter.synopsis}
            Educational points: {', '.join(finished_chapter.educational_points[:3]) if finished_chapter.educational_points else 'general learning'}

            Generate ONE of these bridge content types to keep the child engaged while waiting (in {lang_style['name']}):

            Option A - RECAP: "That was exciting! Remember when [key moment from chapter]?"
            Option B - DID YOU KNOW: "Did you know that [interesting educational fact related to chapter]?"
            Option C - REFLECTION: "What do you think will happen next? Will [character] succeed?"

            Keep it to 2-3 sentences in {lang_style['name']}. Sound natural and engaged.
            End with something like "The next chapter will be ready in just a moment!" (in {lang_style['name']})

            Return ONLY the bridge content text in {lang_style['name']}.""",
            agent=self.dialogue_agent,
            expected_output="Brief engaging bridge content (2-3 sentences)"
        )

        crew = Crew(
            agents=[self.dialogue_agent],
            tasks=[bridge_task],
            process=Process.sequential,
            verbose=False
        )

        result = await self._run_crew_async(crew)
        bridge_text = str(result).strip()

        # Generate audio for bridge content (dialogue = bridge narration)
        from src.services.voice import voice_service
        tts_language = get_tts_language_code(story_language)
        audio_bytes = await voice_service.text_to_speech(
            text=bridge_text,
            speaking_rate=0.9,
            language_code=tts_language,
            use_case="dialogue"
        )

        audio_base64 = None
        if audio_bytes:
            audio_base64 = voice_service.encode_audio_base64(audio_bytes)

        return {
            "type": "bridge",
            "message": bridge_text,
            "audio": audio_base64
        }

    # =========================================================================
    # Voice Direction (Audiobook Production)
    # =========================================================================

    def _strip_markdown(self, text: str) -> str:
        """
        Strip markdown formatting from prose text.

        This is a preprocessing step before voice direction - the VoiceDirectorAgent
        should receive clean prose without markdown artifacts.

        Handles:
        - Bold (**text** or __text__)
        - Italic (*text* or _text_)
        - Headers (# ## ### etc.)
        - Lists (- or * or 1.)
        - Blockquotes (>)
        - Code blocks (``` or `)
        - Links [text](url)

        Preserves:
        - Paragraph breaks
        - Dialogue quotes
        - Punctuation
        """
        import re

        # Remove bold
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)

        # Remove italic (careful not to touch asterisks in other contexts)
        text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)
        text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'\1', text)

        # Remove headers (lines starting with #)
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

        # Remove blockquotes
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

        # Remove links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

        # Remove inline code
        text = re.sub(r'`([^`]+)`', r'\1', text)

        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)

        # Clean up extra whitespace but preserve paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)

        return text.strip()

    async def _auto_generate_audio_tags(self, story_id: str, chapter_number: int):
        """
        Auto-generate ElevenLabs audio tags in background after chapter is written.

        Uses VoiceDirectorAgent to add [emotion] tags like [whispers], [excited], etc.
        These tags are interpreted by ElevenLabs Eleven v3 for expressive narration.

        This is a cheap LLM call (~$0.01) that prepares the chapter for TTS.
        The actual audio generation (expensive) is triggered separately by user action.

        Now emits progress events and updates tts_status for visibility into the process.
        """
        try:
            print(f"\n🎭 Auto-generating audio tags for Chapter {chapter_number}...")

            # Emit starting event
            await story_events.emit("audio_tags_generating", story_id, {
                "chapter_number": chapter_number,
                "status": "generating"
            })

            # Update chapter tts_status to generating
            await self._update_chapter_tts_status(story_id, chapter_number, "generating")

            result = await self.voice_direct_chapter(story_id, chapter_number)

            if result.get("success"):
                print(f"✅ Audio tags ready for Chapter {chapter_number} ({result.get('tts_content_length', 0)} chars)")

                # Update chapter tts_status to ready
                await self._update_chapter_tts_status(story_id, chapter_number, "ready")

                # Emit success event so frontend can update UI
                await story_events.emit("audio_tags_ready", story_id, {
                    "chapter_number": chapter_number,
                    "tts_content_length": result.get("tts_content_length", 0),
                    "estimated_duration_seconds": result.get("estimated_duration_seconds", 0),
                    "status": "ready"
                })
            elif result.get("skipped"):
                print(f"⏭️ Audio tags already exist for Chapter {chapter_number}, skipping")
                await story_events.emit("audio_tags_ready", story_id, {
                    "chapter_number": chapter_number,
                    "status": "ready",
                    "skipped": True
                })
            else:
                error_msg = result.get('error', 'unknown')
                print(f"⚠️ Audio tag generation returned: {error_msg}")

                # Update chapter tts_status to failed
                await self._update_chapter_tts_status(story_id, chapter_number, "failed", error_msg)

                await story_events.emit("audio_tags_failed", story_id, {
                    "chapter_number": chapter_number,
                    "error": error_msg,
                    "status": "failed"
                })

        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ Auto audio tag generation failed for Chapter {chapter_number}: {error_msg}")

            # Update chapter tts_status to failed
            await self._update_chapter_tts_status(story_id, chapter_number, "failed", error_msg)

            # Emit failure event
            await story_events.emit("audio_tags_failed", story_id, {
                "chapter_number": chapter_number,
                "error": error_msg,
                "status": "failed"
            })
            # Don't raise - this is background work

    async def _update_chapter_tts_status(
        self,
        story_id: str,
        chapter_number: int,
        status: str,
        error: str = None
    ):
        """Update the tts_status field for a specific chapter."""
        try:
            story = await self.storage.get_story(story_id)
            if not story:
                return

            for chapter in story.chapters:
                if chapter.number == chapter_number:
                    chapter.tts_status = status
                    if error:
                        chapter.tts_error = error
                    elif status == "ready":
                        chapter.tts_error = None
                    break

            await self.storage.update_story(story_id, story)
        except Exception as e:
            self.logger.error("TTSStatusUpdate", f"Failed to update tts_status: {e}")

    async def voice_direct_chapter(
        self,
        story_id: str,
        chapter_number: int,
        force_regenerate: bool = False
    ) -> Dict[str, Any]:
        """
        Apply voice direction to a written chapter.

        This is a SEPARATE process from chapter writing - like audiobook production
        where a different team handles how the story is voiced vs. how it's written.

        Can be triggered:
        1. Manually via API endpoint
        2. In batch for all chapters

        Args:
            story_id: Story ID
            chapter_number: Chapter to voice direct
            force_regenerate: If True, regenerate even if tts_content exists

        Returns:
            Dict with success status, tts_content length, and metadata counts
        """
        print(f"\n🎭 Voice Direction: Processing Chapter {chapter_number} for story {story_id}")

        # Check if voice direction is enabled
        from src.config import get_settings
        settings = get_settings()
        if not settings.voice_direction_enabled:
            return {
                "success": False,
                "error": "Voice direction is disabled. Set voice_direction_enabled=True in settings."
            }

        # Get story and chapter
        story = await self.storage.get_story(story_id)
        if not story:
            return {"success": False, "error": f"Story {story_id} not found"}

        chapter = None
        for ch in story.chapters:
            if ch.number == chapter_number:
                chapter = ch
                break

        if not chapter:
            return {"success": False, "error": f"Chapter {chapter_number} not found in story"}

        # Check if already processed
        if chapter.tts_content and not force_regenerate:
            return {
                "success": True,
                "message": "Voice direction already exists",
                "tts_content_length": len(chapter.tts_content),
                "skipped": True
            }

        # Get characters for voice mapping
        characters = story.characters or []

        # Get target_age from story preferences (grounded from child profile)
        target_age = 10  # Default
        if story.preferences and story.preferences.target_age:
            target_age = story.preferences.target_age
        elif story.preferences:
            # Fallback: derive from difficulty level if target_age not set
            difficulty = getattr(story.preferences, 'difficulty', None)
            difficulty_str = str(difficulty.value if hasattr(difficulty, 'value') else difficulty).lower()
            if difficulty_str == 'easy':
                target_age = 6
            elif difficulty_str == 'hard':
                target_age = 12

        # Strip markdown from chapter content
        clean_content = self._strip_markdown(chapter.content)
        print(f"   📝 Stripped markdown: {len(chapter.content)} -> {len(clean_content)} chars")

        # Prepare task description
        character_names = [c.name for c in characters]
        task_description = f"""Voice direct Chapter {chapter_number}: "{chapter.title}"

TARGET AGE: {target_age}

CHARACTERS IN STORY:
{', '.join(character_names) if character_names else 'No characters defined yet'}

CHAPTER CONTENT (markdown stripped):
{clean_content}

Transform this chapter into expressive narration using ElevenLabs audio tags.

AUDIO TAGS TO USE (place BEFORE the text they modify):
- Emotions: [whispers], [laughing], [excited], [sad], [angry], [scared], [surprised]
- Delivery: [cautiously], [cheerfully], [nervously], [dramatically], [tenderly], [urgently], [mysteriously], [thoughtfully]
- Effects (use sparingly): [sighing], [gasping], [giggling], [groaning]

RULES:
- NEVER change the actual words - only ADD [tags] in square brackets
- Place tags BEFORE the text they modify: [excited] "I found it!"
- 2-4 tags per paragraph is usually enough - don't over-tag
- Dialogue almost always needs an emotional tag
- Use ellipses (...) for trailing off, dashes (-) for interruptions
- Calculate estimated duration at ~150 words per minute

Output ONLY valid JSON with this structure:
{{
    "tts_content": "The complete chapter text with [audio tags] inserted at appropriate points...",
    "character_emotions": {{
        "CharacterName": ["excited", "whispers", "surprised"]
    }},
    "scene_mood": "Brief description of overall chapter mood",
    "estimated_duration_seconds": N
}}
"""

        # Execute voice direction task
        voice_task = Task(
            description=task_description,
            agent=self.voice_director_agent,
            expected_output="Valid JSON with tts_content, character_emotions, scene_mood, and estimated_duration_seconds"
        )

        crew = Crew(
            agents=[self.voice_director_agent],
            tasks=[voice_task],
            process=Process.sequential,
            verbose=True
        )

        # === OBSERVATORY: Emit agent_started for VoiceDirectorAgent ===
        await story_events.emit_agent_started(
            story_id, "VoiceDirectorAgent", f"Generating audio tags for Chapter {chapter_number}"
        )

        # Set up model tracking for VoiceDirectorAgent
        tracker = ModelTracker.get_instance()
        configured_model = self.router.get_model_for_agent("voice_director")
        tracker.set_current_agent(f"VoiceDirectorAgent_Ch{chapter_number}", configured_model=configured_model)

        print(f"   🎬 Running VoiceDirectorAgent...")
        result = await self._run_crew_async(crew)
        print(f"   ✅ VoiceDirectorAgent complete")

        # Get actual model used from tracker (captures model-router selection)
        voice_model = tracker.get_last_model(f"VoiceDirectorAgent_Ch{chapter_number}")

        # Parse result
        # Use strict=False to handle unescaped control characters (newlines) in long tts_content
        try:
            result_str = clean_json_output(str(result))
            result_data = json.loads(result_str, strict=False)

            # ElevenLabs audio tags format (no SSML)
            tts_content = result_data.get("tts_content", "")
            character_emotions = result_data.get("character_emotions", {})
            scene_mood = result_data.get("scene_mood", "")
            estimated_duration = result_data.get("estimated_duration_seconds", 0)

            # Update chapter with voice direction (ElevenLabs audio tags, not SSML)
            chapter.tts_content = tts_content

            # Convert character_emotions dict to character voice mappings for metadata
            character_voice_mappings = [
                CharacterVoiceMapping(
                    character_name=name,
                    pitch_adjustment="0%",
                    rate_adjustment="medium",
                    voice_quality=", ".join(emotions) if isinstance(emotions, list) else str(emotions)
                )
                for name, emotions in character_emotions.items()
            ]

            chapter.voice_direction_metadata = VoiceDirectionMetadata(
                character_voice_mappings=character_voice_mappings,
                emotional_beats=[],  # Not used in ElevenLabs format
                target_age=target_age,
                total_estimated_duration_seconds=estimated_duration
            )

            # Save updated chapter
            await self.storage.save_chapter(story_id, chapter)
            print(f"   💾 Saved voice direction to Firebase")

            # === OBSERVATORY: Emit agent_completed for VoiceDirectorAgent ===
            await story_events.emit_agent_completed(
                story_id,
                "VoiceDirectorAgent",
                estimated_duration,  # duration_seconds
                True,  # success
                f"Generated {len(tts_content)} chars, {len(character_emotions)} character emotions",
                model=voice_model  # Actual model from ModelTracker (e.g., "gpt-5-mini-2025-08-07")
            )

            # === OBSERVATORY: Emit voice direction specific event ===
            await story_events.emit("audio_tags_ready", story_id, {
                "chapter_number": chapter_number,
                "tts_content_length": len(tts_content),
                "character_emotions": list(character_emotions.keys()),
                "scene_mood": scene_mood,
                "estimated_duration": estimated_duration
            })

            return {
                "success": True,
                "chapter_number": chapter_number,
                "tts_content_length": len(tts_content),
                "character_emotions_mapped": len(character_emotions),
                "scene_mood": scene_mood,
                "estimated_duration_seconds": estimated_duration
            }

        except json.JSONDecodeError as e:
            print(f"   ❌ Failed to parse voice direction output: {e}")
            # === OBSERVATORY: Emit agent_completed (failure) ===
            await story_events.emit_agent_completed(
                story_id, "VoiceDirectorAgent", 0, False, f"JSON parse error: {e}",
                model=voice_model
            )
            return {
                "success": False,
                "error": f"Failed to parse voice direction JSON: {e}",
                "raw_output": str(result)[:500]
            }
        except Exception as e:
            print(f"   ❌ Voice direction failed: {e}")
            # === OBSERVATORY: Emit agent_completed (failure) ===
            await story_events.emit_agent_completed(
                story_id, "VoiceDirectorAgent", 0, False, str(e),
                model=voice_model
            )
            return {
                "success": False,
                "error": f"Voice direction failed: {e}"
            }
