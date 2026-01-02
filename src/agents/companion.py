"""
CompanionAgent - Always-Available Front-Facing Agent

This agent keeps children engaged during story generation wait times.
Unlike CrewAI agents, this is a lightweight Python class that is never blocked.

Key features:
- Uses Microsoft Foundry Model Router for intelligent model selection
- Fast "balanced" mode for quick responses, "quality" mode for character spotlights
- Proactively sends educational teasers during Chapter 1 wait
- Subscribes to story events for announcements
- Maintains Hanan al-Hroub persona (warm, enthusiastic teacher)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from pathlib import Path

from src.services.events import story_events, StoryEvent
from src.services.voice import voice_service
from src.services.llm_router import get_llm_router
from src.models import InputTier
from src.utils.language_styles import get_language_style, get_dialogue_instruction, get_tts_language_code

logger = logging.getLogger(__name__)


@dataclass
class CompanionState:
    """Tracks CompanionAgent engagement for a single story"""
    story_id: str
    started_at: datetime = field(default_factory=datetime.now)
    announced_characters: Set[str] = field(default_factory=set)
    announced_structure: bool = False
    proactive_teasers_sent: int = 0
    last_proactive_time: Optional[datetime] = None
    educational_topics_covered: Set[str] = field(default_factory=set)
    chapter_1_ready: bool = False
    active: bool = True
    _scheduler_task: Optional[asyncio.Task] = None
    # One-chapter-at-a-time: Track prefetched chapters to avoid duplicates
    prefetching_chapters: Set[int] = field(default_factory=set)
    prefetched_chapters: Set[int] = field(default_factory=set)
    # Chapter playback flow: waiting for user to say "yes" to hear chapter
    pending_chapter_playback: Optional[int] = None
    pending_chapter_title: Optional[str] = None


@dataclass
class PlaybackContext:
    """
    Rich context for CompanionAgent during story playback.

    Unlike CompanionState (which tracks engagement), this provides
    FULL story content for grounded, contextual responses.
    Fetched on-demand when needed for quality responses.
    """
    story_id: str
    current_chapter: int
    playback_phase: str  # Maps to PlaybackPhase enum value

    # Full chapter content (not truncated)
    current_chapter_content: Optional[str] = None
    current_chapter_title: Optional[str] = None

    # Educational elements from the chapter
    vocabulary_words: List[Dict] = field(default_factory=list)
    educational_points: List[str] = field(default_factory=list)

    # Character information (ALL characters, not just first 4)
    characters_featured: List[str] = field(default_factory=list)  # In current chapter
    all_characters: List[Dict] = field(default_factory=list)  # Full character list

    # Story continuity
    previous_chapter_synopsis: Optional[str] = None
    chapters_completed: int = 0
    total_chapters: int = 0

    # Child profile for personalization
    child_age: Optional[int] = None
    child_name: Optional[str] = None


@dataclass
class DialogueResponse:
    """Response from CompanionAgent with text and optional audio."""
    text: str
    audio_base64: Optional[str] = None
    intent_detected: Optional[str] = None  # For logging/debugging


class ContextPage:
    """
    Types of context pages that can be loaded on-demand.

    Used by PaginatedStoryContext to load only what's needed for a given question,
    avoiding the cost of loading the full story context (10K+ tokens) for every query.
    """
    OVERVIEW = "overview"        # Title, theme, total chapters (~50 tokens)
    CHARACTERS = "characters"    # All character arcs and evolution (~500 tokens)
    PLOT_THREADS = "plot_threads"  # All PlotElements with resolutions (~300 tokens)
    CHAPTERS = "chapters"        # All chapter synopses (~400 tokens)
    FULL = "full"               # Everything (expensive, ~2000+ tokens)


@dataclass
class PaginatedStoryContext:
    """
    Lazy-loaded story context for memory efficiency.

    Loads context pages on-demand based on the question asked, reducing token usage
    while maintaining the ability to answer comprehensive questions about the full story.

    Usage:
        context = PaginatedStoryContext(story_id="...", firebase=firebase_service)
        overview = await context.get_page(ContextPage.OVERVIEW)
        characters = await context.get_page(ContextPage.CHARACTERS)
    """
    story_id: str
    firebase: Any = None  # FirebaseService, using Any to avoid circular import
    _story: Optional[Any] = None  # Cached Story object
    _loaded_pages: Dict[str, Any] = field(default_factory=dict)

    async def get_page(self, page: str) -> Dict:
        """
        Load a context page on-demand.

        Pages are cached after first load for the lifetime of this object.

        Args:
            page: One of ContextPage constants (OVERVIEW, CHARACTERS, etc.)

        Returns:
            Dict with the requested context data
        """
        if page in self._loaded_pages:
            return self._loaded_pages[page]

        if self._story is None and self.firebase:
            self._story = await self.firebase.get_story(self.story_id)

        if not self._story:
            return {"error": "Story not found"}

        result = {}

        if page == ContextPage.OVERVIEW:
            result = {
                "title": self._story.structure.title if self._story.structure else "Unknown",
                "theme": self._story.structure.theme if self._story.structure else "Unknown",
                "total_chapters": len(self._story.structure.chapters) if self._story.structure else 0,
                "chapters_completed": len([c for c in (self._story.chapters or []) if c.status == "completed" or c.status == "ready"]),
                "language": self._story.preferences.language if self._story.preferences else "en"
            }

        elif page == ContextPage.CHARACTERS:
            result = {
                "characters": [
                    {
                        "name": c.name,
                        "role": c.role,
                        "arc_summary": self._summarize_character_arc(c),
                        "final_state": {
                            "personality": c.personality_traits[:5] if c.personality_traits else [],
                            "relationships": c.relationships or {},
                            "skills_learned": [s.skill_name for s in (c.progression.skills_learned if c.progression else [])]
                        }
                    }
                    for c in (self._story.characters or [])
                ]
            }

        elif page == ContextPage.PLOT_THREADS:
            plot_elements = []
            if self._story.structure and hasattr(self._story.structure, 'plot_elements'):
                for pe in (self._story.structure.plot_elements or []):
                    plot_elements.append({
                        "name": pe.name,
                        "type": pe.element_type.value if hasattr(pe.element_type, 'value') else str(pe.element_type),
                        "introduced": pe.introduced_chapter,
                        "resolved": pe.resolution_chapter,
                        "status": pe.status
                    })
            result = {"plot_elements": plot_elements}

        elif page == ContextPage.CHAPTERS:
            result = {
                "chapters": [
                    {
                        "number": ch.number,
                        "title": ch.title,
                        "synopsis": (ch.synopsis or "")[:300]
                    }
                    for ch in sorted(self._story.chapters or [], key=lambda x: x.number)
                ]
            }

        self._loaded_pages[page] = result
        return result

    def _summarize_character_arc(self, character) -> str:
        """Generate 1-sentence character arc summary."""
        if not character.progression or not character.progression.personality_evolution:
            return f"{character.name} maintained their {character.role} role throughout."

        evolutions = character.progression.personality_evolution
        if not evolutions:
            return f"{character.name} grew as a {character.role} throughout the story."

        first_evo = evolutions[0]
        last_evo = evolutions[-1] if len(evolutions) > 1 else first_evo
        return f"{character.name} evolved from {first_evo.from_trait} to {last_evo.to_trait}."


# Global instance (set by main.py)
_companion_agent: Optional['CompanionAgent'] = None


def set_companion_agent(agent: 'CompanionAgent'):
    """Set the global companion agent instance"""
    global _companion_agent
    _companion_agent = agent


def get_companion_agent() -> Optional['CompanionAgent']:
    """Get the global companion agent instance"""
    return _companion_agent


class CompanionAgent:
    """
    Always-available front-facing agent for child engagement.

    This agent is NOT a CrewAI Agent - it's a lightweight Python class
    that runs independently and is never blocked by story generation.
    """

    def __init__(
        self,
        firebase_service,
        foundry_endpoint: Optional[str] = None,
        foundry_api_key: Optional[str] = None,
        use_foundry: bool = False,
        proactive_interval: int = 60,  # Increased from 30s to reduce message spam
        max_teasers: int = 3  # Reduced from 4 to keep engagement focused
    ):
        """
        Initialize CompanionAgent.

        Args:
            firebase_service: Firebase service for reading story state
            foundry_endpoint: Azure AI Foundry endpoint URL
            foundry_api_key: Foundry API key (None for Managed Identity)
            use_foundry: Feature flag to enable Foundry (vs legacy direct APIs)
            proactive_interval: Seconds between proactive teasers
            max_teasers: Maximum teasers before Chapter 1
        """
        self.firebase = firebase_service
        self.proactive_interval = proactive_interval
        self.max_teasers = max_teasers

        # Initialize Foundry client for intelligent model routing
        self._foundry_service = None
        self._use_foundry = use_foundry

        if use_foundry and foundry_endpoint:
            from src.services.foundry import FoundryService
            self._foundry_service = FoundryService(
                endpoint=foundry_endpoint,
                api_key=foundry_api_key,
                routing_mode="balanced"  # Default to balanced for fast responses
            )
            logger.info("CompanionAgent initialized with Foundry Model Router")
        else:
            logger.warning("CompanionAgent: Foundry not configured - responses will fail!")

        # Per-story engagement state
        self._states: Dict[str, CompanionState] = {}

        # Load Hanan persona prompt
        self._persona_prompt = self._load_persona_prompt()

        # Subscribe to events
        self._subscribe_to_events()

        logger.info("CompanionAgent initialized")

    def _load_persona_prompt(self) -> str:
        """Load the Hanan persona prompt from file"""
        prompt_path = Path(__file__).parent.parent / "prompts" / "companion_agent.txt"
        try:
            with open(prompt_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            # Fallback to inline prompt if file doesn't exist yet
            return self._get_fallback_persona()

    def _get_fallback_persona(self) -> str:
        """Fallback Hanan persona if prompt file is missing"""
        return """You are Hanan, a warm and enthusiastic teacher who LOVES stories.
You embody Hanan al-Hroub's "We Play, We Learn" philosophy.

Your personality:
- Warm, encouraging, like a favorite teacher
- Genuinely excited about stories and learning
- Patient and never rushes the child
- Speaks in 2-3 sentence bursts
- Uses simple, age-appropriate language

You celebrate children's ideas and make them feel like creative partners.
Every child is a "true winner" in their learning journey."""

    def _subscribe_to_events(self):
        """Subscribe to story events for proactive announcements"""
        story_events.on("structure_ready", self._on_structure_ready)
        story_events.on("character_ready", self._on_character_ready)
        story_events.on("chapter_ready", self._on_chapter_ready)
        logger.info("CompanionAgent subscribed to story events")

    async def activate(self, story_id: str, story_context: Optional[Dict] = None):
        """
        Activate CompanionAgent for a story.
        Called when background generation starts.

        Args:
            story_id: The story ID
            story_context: Optional initial context (theme, prompt, etc.)
        """
        if story_id in self._states:
            logger.warning(f"CompanionAgent already active for {story_id}")
            return

        state = CompanionState(story_id=story_id)
        self._states[story_id] = state

        # Start proactive teaser scheduler
        state._scheduler_task = asyncio.create_task(
            self._proactive_scheduler(story_id, story_context)
        )

        logger.info(f"CompanionAgent activated for story {story_id}")

    async def deactivate(self, story_id: str):
        """
        Deactivate proactive engagement (called when Chapter 1 is ready).
        The agent still handles user messages.

        Args:
            story_id: The story ID
        """
        if story_id not in self._states:
            return

        state = self._states[story_id]
        state.active = False
        state.chapter_1_ready = True

        # Cancel scheduler task
        if state._scheduler_task:
            state._scheduler_task.cancel()
            try:
                await state._scheduler_task
            except asyncio.CancelledError:
                pass

        logger.info(f"CompanionAgent deactivated proactive engagement for {story_id}")

    def cleanup(self, story_id: str):
        """
        Fully cleanup state for a story (when WebSocket disconnects).

        Args:
            story_id: The story ID
        """
        if story_id in self._states:
            state = self._states[story_id]
            if state._scheduler_task:
                state._scheduler_task.cancel()
            del self._states[story_id]
            logger.info(f"CompanionAgent cleaned up for story {story_id}")

    # ===== Conversation-First Greeting =====

    async def generate_greeting(
        self,
        child_name: str,
        message: str,
        intent: str,  # "continue", "new_story", "exploring", "greeting"
        active_story: Optional[Any],
        language: str = "en",
        child_age: int = 7,
        conversation_turn: int = 1,
        story_library: Optional[List[Dict]] = None
    ) -> DialogueResponse:
        """
        Generate context-aware greeting based on child's intent.

        This is the entry point for conversation-first UX where
        CompanionAgent engages BEFORE story generation starts.

        Args:
            child_name: Child's name for personalization
            message: The child's message
            intent: Classified intent (continue, new_story, exploring, greeting)
            active_story: The child's active story if one exists
            language: Language code (en, no, es)
            child_age: Child's age for age-appropriate responses
            conversation_turn: Current turn in exploring mode (1-3)
            story_library: List of all child's stories with summaries

        Returns:
            DialogueResponse with greeting text and optional audio
        """
        lang_style = get_language_style(language)
        lang_instruction = get_dialogue_instruction(language)

        # Build story library context string
        story_library = story_library or []
        has_multiple_stories = len(story_library) > 1

        story_list_str = ""
        if story_library:
            story_list_str = "\n".join([
                f"- \"{s['title']}\" ({s['theme'] or 'adventure'}){' ← ACTIVE' if s.get('is_active') else ''}"
                for s in story_library
            ])

        # Build context-aware prompt based on intent
        if intent == "continue":
            # Get story details for a warm welcome back
            # NOTE: Story title is in story.structure.title, not story.title
            story_title = "your story"
            if active_story and active_story.structure:
                story_title = active_story.structure.title

            # Try to get current chapter info
            current_chapter = 1
            chapter_title = None
            if active_story:
                if hasattr(active_story, 'reading_state') and active_story.reading_state:
                    current_chapter = getattr(active_story.reading_state, 'current_chapter', 1)
                # Get chapter title from structure.chapters first (always populated)
                if active_story.structure and active_story.structure.chapters:
                    for ch in active_story.structure.chapters:
                        if ch.number == current_chapter:
                            chapter_title = ch.title
                            break
                # Fall back to story.chapters (populated after chapter is written)
                if not chapter_title and hasattr(active_story, 'chapters') and active_story.chapters:
                    for ch in active_story.chapters:
                        if ch.number == current_chapter:
                            chapter_title = ch.title
                            break
            # Default fallback
            if not chapter_title:
                chapter_title = f"Chapter {current_chapter}"

            # If multiple stories, ask which one to continue
            if has_multiple_stories:
                prompt = f"""You are Hanan, a warm storyteller helping a child choose which story to continue.

LANGUAGE: Respond in {lang_style['name']}.
{lang_instruction}

Child's name: {child_name}
Child's age: {child_age}
Child said: "{message}"

THEIR STORY LIBRARY:
{story_list_str}

The child wants to continue but has multiple stories!
Generate a warm response (2-3 sentences):
1. Greet them warmly by name
2. Acknowledge they have several adventures to choose from
3. List 2-3 of their stories by name and ask which one they'd like to continue
   (The one marked ACTIVE was their most recent)

Keep it simple and warm - don't overwhelm them with choices.
Return ONLY the greeting text in {lang_style['name']}."""
            elif active_story:
                prompt = f"""You are Hanan, a storyteller about to start the story.

LANGUAGE: Respond in {lang_style['name']}.
{lang_instruction}

Story to continue: "{story_title}"

The child ALREADY said they want to continue. This is NOT a greeting moment!
Generate a VERY brief action confirmation (1 sentence ONLY):
- Start with "Flott!" or "Great!" or "Perfekt!"
- Say we're starting/continuing the story NOW
- Do NOT say the child's name (they were already greeted!)
- Do NOT say "how nice to see you" or any welcome phrases

EXAMPLES (pick similar style):
- "Flott! La oss fortsette eventyret!"
- "Perfekt! Da setter vi i gang med historien!"
- "Great! Let's continue the story!"

ONE SENTENCE MAXIMUM. No greetings. Just action.
Return ONLY the confirmation text in {lang_style['name']}."""
            else:
                # No stories to continue - guide them to start one
                prompt = f"""You are Hanan, a warm storyteller greeting a child.

LANGUAGE: Respond in {lang_style['name']}.
{lang_instruction}

Child's name: {child_name}
Child's age: {child_age}
Child said: "{message}"

The child wants to continue a story, but they don't have any stories yet!
Generate a warm, encouraging response (2-3 sentences):
1. Greet them warmly
2. Gently let them know they don't have a story to continue yet (without making them feel bad)
3. Excitedly suggest starting their first adventure together

Keep it warm and positive.
Return ONLY the greeting text in {lang_style['name']}."""

        elif intent == "new_story":
            # Extract topic if mentioned
            prompt = f"""You are Hanan, a warm storyteller responding to a new story request.

LANGUAGE: Respond in {lang_style['name']}.
{lang_instruction}

Child's name: {child_name}
Child's age: {child_age}
Child said: "{message}"

Generate an enthusiastic response (2-3 sentences):
1. Show excitement about their topic/idea
2. Mention one interesting fact related to their topic (if applicable)
3. If you need more info, ask ONE clarifying question (e.g., scary or funny?)
   BUT if you have enough context, make an excited suggestion instead!
   Do NOT always end with a question - vary your approach.

Keep it age-appropriate and genuine.
Return ONLY the response text in {lang_style['name']}."""

        elif intent == "exploring":
            # Help them discover what they want
            # Build story options from library
            existing_story_options = ""
            if story_library:
                story_names = [f'"{s["title"]}"' for s in story_library[:3]]  # Max 3
                if len(story_names) == 1:
                    existing_story_options = f"- Continue their story {story_names[0]}"
                elif story_names:
                    existing_story_options = f"- Continue one of their stories: {', '.join(story_names)}"

            # After 2-3 turns, gently suggest a topic
            if conversation_turn >= 3:
                prompt = f"""You are Hanan, helping a young child choose a story.

LANGUAGE: Respond in {lang_style['name']}.
{lang_instruction}

Child's name: {child_name}
Child's age: {child_age}
Child said: "{message}"
Conversation turn: {conversation_turn} (child has been exploring for a while)

{f"THEIR EXISTING STORIES:{chr(10)}{story_list_str}" if story_library else "They don't have any stories yet."}

Since they've been exploring for a bit, gently suggest something specific.
Generate a warm response (2-3 sentences):
1. Acknowledge what they said
2. Either suggest continuing one of their existing stories (if they have any)
   OR suggest a specific new story idea based on common interests for their age
   (e.g., "How about a story about a brave little bear who goes on an adventure?")
3. Ask if that sounds good

Keep it simple and encouraging.
Return ONLY the response text in {lang_style['name']}."""
            else:
                prompt = f"""You are Hanan, helping a young child decide what they want.

LANGUAGE: Respond in {lang_style['name']}.
{lang_instruction}

Child's name: {child_name}
Child's age: {child_age}
Child said: "{message}"

They seem unsure about what story they want.

{f"THEIR STORY LIBRARY (they can continue any of these!):{chr(10)}{story_list_str}" if story_library else "They don't have any stories yet - this will be their first adventure!"}

Generate a warm, helpful response (2-3 sentences):
1. Greet them warmly (if first turn) or acknowledge their message
2. Offer 2-3 simple options - INCLUDE their existing stories if they have any:
   {existing_story_options}
   - Start a new adventure (suggest age-appropriate themes)
   - For ages 3-5: "dragons or cute animals?" / "princess or pirate?"
   - For ages 6-9: "adventure, mystery, or funny story?"
   - For ages 10+: More open-ended suggestions
3. Make it feel like a fun choice, not a test

Keep it simple and encouraging.
Return ONLY the response text in {lang_style['name']}."""

        else:  # greeting
            prompt = f"""You are Hanan, a warm storyteller greeting a child.

LANGUAGE: Respond in {lang_style['name']}.
{lang_instruction}

Child's name: {child_name}
Child's age: {child_age}
Child said: "{message}"

{f"THEIR STORY LIBRARY:{chr(10)}{story_list_str}" if story_library else "This is their first visit - no stories yet!"}

Generate a warm greeting (2-3 sentences):
1. Greet them warmly by name
2. Express excitement to see them
3. If they have existing stories, mention them and ask if they'd like to continue one OR start something new
   If no stories yet, ask what kind of adventure they'd like to begin

Keep it warm, genuine, and age-appropriate.
Return ONLY the greeting text in {lang_style['name']}."""

        # Generate response using Foundry Model Router
        try:
            response_text = await self._generate_fast_response(prompt)

            # Generate audio (dialogue = companion greeting)
            tts_language = get_tts_language_code(language)
            audio_bytes = await voice_service.text_to_speech(
                text=response_text,
                speaking_rate=0.9,
                language_code=tts_language,
                use_case="dialogue"
            )

            audio_base64 = None
            if audio_bytes:
                audio_base64 = voice_service.encode_audio_base64(audio_bytes)

            logger.info(f"CompanionAgent greeting generated for {child_name} (intent: {intent}, lang: {language})")

            return DialogueResponse(
                text=response_text,
                audio_base64=audio_base64,
                intent_detected=intent
            )

        except Exception as e:
            logger.error(f"Error generating greeting: {e}")
            # Fallback responses by language and intent
            fallbacks = {
                "en": {
                    "continue": f"Great, {child_name}! Let's continue your story!",
                    "new_story": f"That sounds exciting, {child_name}! What kind of story would you like?",
                    "exploring": f"Hi {child_name}! Would you like to hear about dragons, animals, or an adventure?",
                    "greeting": f"Hello {child_name}! I'm so happy to see you! What story shall we explore today?"
                },
                "no": {
                    "continue": f"Flott, {child_name}! La oss fortsette historien din!",
                    "new_story": f"Det høres spennende ut, {child_name}! Hva slags historie vil du ha?",
                    "exploring": f"Hei {child_name}! Vil du høre om drager, dyr, eller et eventyr?",
                    "greeting": f"Hei {child_name}! Så hyggelig å se deg! Hvilken historie skal vi utforske i dag?"
                },
                "es": {
                    "continue": f"¡Genial, {child_name}! ¡Continuemos tu historia!",
                    "new_story": f"¡Qué emocionante, {child_name}! ¿Qué tipo de historia te gustaría?",
                    "exploring": f"¡Hola {child_name}! ¿Te gustaría escuchar sobre dragones, animales o una aventura?",
                    "greeting": f"¡Hola {child_name}! ¡Qué alegría verte! ¿Qué historia exploraremos hoy?"
                }
            }

            lang_fallbacks = fallbacks.get(language, fallbacks["en"])
            fallback_text = lang_fallbacks.get(intent, lang_fallbacks["greeting"])

            return DialogueResponse(
                text=fallback_text,
                audio_base64=None,
                intent_detected=intent
            )

    async def _generate_fast_response(self, prompt: str) -> str:
        """Generate fast response using Foundry Model Router with balanced mode and low reasoning."""
        if not self._foundry_service:
            raise RuntimeError("Foundry service not initialized")

        response = await self._foundry_service.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            routing_mode="balanced",  # Fast, cost-effective for greetings
            max_tokens=1500,  # Reasoning models use 500-1000 internal tokens before output
            temperature=0.7,
            reasoning_effort="low"  # Minimize thinking tokens for faster dialogue
        )

        logger.debug(f"Fast response used model: {response.get('model', 'unknown')}")
        return response["content"].strip()

    # ===== User Message Handling =====

    async def handle_user_message(
        self,
        story_id: str,
        user_message: str,
        tier: InputTier,
        classification: Any,
        story: Any,
        chapter_triggering: bool = False,
        chapter_number: Optional[int] = None
    ) -> str:
        """
        Handle user message with Hanan persona.

        Args:
            story_id: The story ID
            user_message: The user's message
            tier: Input tier classification
            classification: Full classification object
            story: Story object from Firebase
            chapter_triggering: If True, chapter generation is about to start (respond with excitement!)
            chapter_number: Optional chapter number being generated (for continuation)

        Returns:
            Response message (audio generated separately)
        """
        # Emit agent_started event for Observatory (no model - we don't know it until response)
        await story_events.emit(
            "agent_started", story_id,
            {"agent": "CompanionAgent", "task": f"Responding to: {user_message[:30]}..."}
        )

        # NOTE: Confirmation gate removed - chapters auto-start now
        # The pending_chapter_playback check is no longer needed

        # Build rich context
        rich_context = self._build_rich_context(story)

        # Extract language from story preferences
        story_language = story.preferences.language if story and story.preferences else "en"

        # ===== CHECK FOR CHAPTER-SPECIFIC QUESTIONS =====
        # Detect questions like "what happens in chapter 2?" or "hva skjer i kapittel 2?"
        # If the chapter isn't generated, trigger its generation
        chapter_num = self._detect_chapter_question(user_message)
        if chapter_num:
            chapter_response = await self._handle_chapter_question(
                story_id, chapter_num, story_language
            )
            if chapter_response:
                # Chapter generation was triggered, return the response
                await story_events.emit(
                    "agent_completed", story_id,
                    {"agent": "CompanionAgent", "success": True, "triggered_chapter": chapter_num}
                )
                return chapter_response
            # If None, chapter exists and we continue with normal handling

        # Different handling based on tier
        try:
            if tier == InputTier.TIER_1_IMMEDIATE:
                result = await self._handle_immediate(
                    story_id, user_message, classification, rich_context, story_language, chapter_triggering
                )
            elif tier in (InputTier.TIER_2_PREFERENCE, InputTier.TIER_3_STORY_CHOICE,
                          InputTier.TIER_4_ADDITION):
                result = await self._handle_influence(
                    story_id, user_message, tier, classification, rich_context, story_language
                )
            else:
                result = await self._handle_immediate(
                    story_id, user_message, classification, rich_context, story_language
                )

            # Emit agent_completed event for Observatory
            await story_events.emit(
                "agent_completed", story_id,
                {"agent": "CompanionAgent", "success": True}
            )
            return result
        except Exception as e:
            # Emit agent_completed with error for Observatory
            await story_events.emit(
                "agent_completed", story_id,
                {"agent": "CompanionAgent", "success": False, "error": str(e)}
            )
            raise

    async def _handle_immediate(
        self,
        story_id: str,
        user_message: str,
        classification: Any,
        rich_context: str,
        language: str = "en",
        chapter_triggering: bool = False
    ) -> str:
        """Handle Tier 1 immediate response"""
        # Get language-specific dialogue instruction
        lang_instruction = get_dialogue_instruction(language)
        lang_style = get_language_style(language)

        # SPECIAL CASE: Chapter is being triggered - respond with excitement!
        if chapter_triggering:
            if chapter_number and chapter_number > 1:
                # Continuation - generating a chapter beyond Chapter 1
                prompt = f"""{self._persona_prompt}

{lang_instruction}

The child wants to CONTINUE the story and Chapter {chapter_number} generation is STARTING NOW!

Child said: "{user_message}"

Generate a SHORT excited response (1-2 sentences MAX) announcing Chapter {chapter_number} is being written:
- Show excitement about continuing the adventure
- Mention Chapter {chapter_number} is starting
- Do NOT ask any questions

Examples:
- Norwegian: "Absolutt! La meg skrive kapittel {chapter_number} med en gang - det blir spennende!"
- English: "Absolutely! Let me write Chapter {chapter_number} right now - it's going to be exciting!"
- Spanish: "¡Por supuesto! ¡Vamos a escribir el capítulo {chapter_number} ahora mismo!"

Return ONLY the dialogue in {lang_style['name']}. Keep it SHORT!"""
            else:
                # Starting Chapter 1
                prompt = f"""{self._persona_prompt}

{lang_instruction}

The child wants to start the story and chapter generation is STARTING NOW!

Child said: "{user_message}"

Generate a SHORT excited response (1-2 sentences MAX) announcing you're starting the story:
- Show excitement
- Say the story is starting NOW
- Do NOT ask any questions

Examples:
- Norwegian: "Fantastisk! La meg begynne på eventyret ditt med en gang!"
- English: "Wonderful! Let me start your story right now!"
- Spanish: "¡Maravilloso! ¡Vamos a empezar tu cuento ahora mismo!"

Return ONLY the dialogue in {lang_style['name']}. Keep it SHORT - the story is about to begin!"""
        else:
            prompt = f"""{self._persona_prompt}

{lang_instruction}

=== CURRENT INTERACTION ===
The child just said: "{user_message}"
Their intent: {getattr(classification, 'classified_intent', 'general question')}

=== STORY CONTEXT ===
{rich_context}
=== END CONTEXT ===

CRITICAL - STORYTELLER ROLE:
- You are the STORYTELLER. You TELL stories TO the child.
- NEVER ask the child to "tell you" the story or "what happens next"
- NEVER say things like "Fortell meg..." (Tell me...) or "Hva tror du skjer?" (What do you think happens?)
- The child is the LISTENER, not the narrator

Respond naturally in 2-3 sentences as Hanan in {lang_style['name']}:
- If they ask about characters: Share SPECIFIC traits from context
- If they ask about the story: Reference SPECIFIC plot points
- If acknowledgment ("cool!", "wow!"): Brief enthusiastic response
- If they ask "what's happening" or status: Reassure them warmly

Be warm, specific, and engaging. Use simple words for a child.
Return ONLY the dialogue text in {lang_style['name']}, nothing else."""

        try:
            # Use LLM Router to respect models.yaml configuration (dialogue: gpt-4o-mini)
            result = await self._call_with_llm_router(prompt)
            return result["content"]  # Extract content, model available in result["model"] if needed
        except Exception as e:
            logger.error(f"CompanionAgent fast response error: {e}")
            # Fallback messages by language
            if chapter_triggering:
                fallbacks = {
                    "no": "Fantastisk! La meg begynne på historien din nå!",
                    "es": "¡Maravilloso! ¡Vamos a empezar tu cuento ahora!",
                    "en": "Wonderful! Let me start your story now!"
                }
            else:
                fallbacks = {
                    "no": "Jeg jobber med historien din akkurat nå! Den blir fantastisk - bare litt til!",
                    "es": "¡Estoy trabajando en tu historia ahora mismo! Va a ser increíble - ¡solo un poco más!",
                    "en": "I'm working on your story right now! It's going to be amazing - just a little more time!"
                }
            return fallbacks.get(language, fallbacks["en"])

    async def _call_foundry(
        self,
        prompt: str,
        routing_mode: str = "balanced",
        max_tokens: int = 1500,
        reasoning_effort: str = "low",
        story_id: str = None
    ) -> str:
        """Call Foundry Model Router for responses.

        Args:
            prompt: The prompt to send
            routing_mode: "balanced" for fast responses, "quality" for better output
            max_tokens: Max response tokens (default 1500 for reasoning models that use ~500-1000 internal tokens)
            reasoning_effort: For reasoning models (gpt-5-mini, etc.) - "low" for fast responses,
                              "medium"/"high" for complex tasks. Default "low" for dialogue.
            story_id: Optional story ID for observatory event emission
        """
        import time
        call_start = time.time()

        if not self._foundry_service:
            raise Exception("Foundry service not initialized")

        # Log the prompt being sent (truncated)
        prompt_preview = prompt[:200].replace('\n', ' ') if len(prompt) > 200 else prompt.replace('\n', ' ')
        logger.info(f"🗣️ CompanionAgent calling Foundry (mode={routing_mode}, max_tokens={max_tokens}, reasoning={reasoning_effort})")
        logger.info(f"   📝 Prompt: {prompt_preview}...")

        response = await self._foundry_service.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            routing_mode=routing_mode,
            max_tokens=max_tokens,
            temperature=0.7,
            reasoning_effort=reasoning_effort  # Reduce internal thinking for faster dialogue
        )

        model_used = response.get('model', 'unknown')
        tokens_used = response.get('usage', {}).get('total_tokens', 0)
        tokens_in = response.get('usage', {}).get('prompt_tokens', 0)
        tokens_out = response.get('usage', {}).get('completion_tokens', 0)
        reasoning_tokens = response.get('usage', {}).get('completion_tokens_details', {})
        if isinstance(reasoning_tokens, dict):
            reasoning_tokens = reasoning_tokens.get('reasoning_tokens', 0)
        else:
            reasoning_tokens = 0
        logger.info(f"🤖 CompanionAgent response: model={model_used}, tokens={tokens_used}, reasoning={reasoning_tokens}")

        # === OBSERVATORY: Emit model routing events ===
        latency_ms = int((time.time() - call_start) * 1000)
        if story_id:
            await story_events.emit_model_selected(
                story_id, "CompanionAgent", model_used, routing_mode
            )
            await story_events.emit_model_response(
                story_id, model_used, tokens_in, tokens_out, latency_ms
            )

        # Return both content and actual model for observatory events
        return {"content": response["content"].strip(), "model": model_used}

    async def _call_with_llm_router(
        self,
        prompt: str,
        max_tokens: int = 4096,
        story_id: str = None
    ) -> dict:
        """Call LLM using configured model from models.yaml via LLM Router.

        This method replaces _call_foundry() for dialogue to ensure CompanionAgent
        respects the models.yaml configuration (e.g., dialogue: gpt-4o-mini).

        Args:
            prompt: The prompt to send
            max_tokens: Max response tokens
            story_id: Optional story ID for observatory event emission

        Returns:
            dict with 'content' and 'model' keys
        """
        import time
        import litellm

        call_start = time.time()

        # Get model configuration from LLM Router
        router = get_llm_router()
        model = router.get_model_for_agent("dialogue")  # Gets model from models.yaml
        config = router.get_llm_kwargs("dialogue")

        # Log the prompt being sent (truncated)
        prompt_preview = prompt[:200].replace('\n', ' ') if len(prompt) > 200 else prompt.replace('\n', ' ')
        logger.info(f"🗣️ CompanionAgent calling LLM Router (model={model})")
        logger.info(f"   📝 Prompt: {prompt_preview}...")

        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.get("temperature", 0.8),
                max_tokens=config.get("max_tokens", max_tokens),
            )

            content = response.choices[0].message.content
            model_used = response.model

            tokens_in = response.usage.prompt_tokens if response.usage else 0
            tokens_out = response.usage.completion_tokens if response.usage else 0
            tokens_total = response.usage.total_tokens if response.usage else 0

            logger.info(f"🤖 CompanionAgent response: model={model_used}, tokens={tokens_total}")

            # === OBSERVATORY: Emit model routing events ===
            latency_ms = int((time.time() - call_start) * 1000)
            if story_id:
                await story_events.emit_model_selected(
                    story_id, "CompanionAgent", model_used, "llm_router"
                )
                await story_events.emit_model_response(
                    story_id, model_used, tokens_in, tokens_out, latency_ms
                )

            return {"content": content.strip() if content else "", "model": model_used}

        except Exception as e:
            logger.error(f"❌ LLM Router call failed: {e}")
            raise

    async def stream_response(
        self,
        story_id: str,
        user_message: str,
        story: Any,
        classification: Any = None,
        chapter_triggering: bool = False,
        chapter_number: Optional[int] = None
    ):
        """
        DEPRECATED: This method doesn't work correctly with Azure Model Router.

        The issue is that litellm.acompletion("azure/model-router", stream=True)
        returns HTTP 200 but yields empty chunks. Azure's Model Router streaming
        format is not properly handled by litellm.

        To fix: Implement streaming in FoundryService using AsyncAzureOpenAI's
        streaming support, then call self._foundry_service.stream_chat_completion().

        For now, use _handle_immediate() instead which works correctly.

        Original docstring:
        Stream dialogue response tokens as they arrive.
        This provides immediate user feedback by yielding text chunks
        as the LLM generates them. Audio is generated after text completes.

        Args:
            story_id: The story ID
            user_message: The user's message
            story: Story object with preferences
            classification: Optional classification object
            chapter_triggering: If True, chapter generation is starting
            chapter_number: Optional chapter number for continuations

        Yields:
            str: Text chunks as they arrive from the LLM
        """
        import litellm

        # Build the prompt
        story_language = story.preferences.language if story and story.preferences else "en"
        lang_instruction = get_dialogue_instruction(story_language)
        lang_style = get_language_style(story_language)
        rich_context = self._build_rich_context(story)

        # Build prompt based on context
        if chapter_triggering:
            if chapter_number and chapter_number > 1:
                prompt = f"""{self._persona_prompt}

{lang_instruction}

The child wants to CONTINUE the story and Chapter {chapter_number} generation is STARTING NOW!

Child said: "{user_message}"

Generate a SHORT excited response (1-2 sentences MAX) announcing Chapter {chapter_number} is being written.
Return ONLY the dialogue in {lang_style['name']}. Keep it SHORT!"""
            else:
                prompt = f"""{self._persona_prompt}

{lang_instruction}

The child wants to start the story and chapter generation is STARTING NOW!

Child said: "{user_message}"

Generate a SHORT excited response (1-2 sentences MAX) announcing you're starting the story.
Return ONLY the dialogue in {lang_style['name']}. Keep it SHORT!"""
        else:
            intent = getattr(classification, 'classified_intent', 'general question') if classification else 'general question'
            prompt = f"""{self._persona_prompt}

{lang_instruction}

=== CURRENT INTERACTION ===
The child just said: "{user_message}"
Their intent: {intent}

=== STORY CONTEXT ===
{rich_context}
=== END CONTEXT ===

Respond naturally in 2-3 sentences as Hanan in {lang_style['name']}.
"""

        logger.info(f"🔄 Streaming response for: {user_message[:50]}...")

        try:
            # Use azure/model-router for streaming
            response = await litellm.acompletion(
                model="azure/model-router",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=500,
                temperature=0.7
            )

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            # Fallback: yield error message
            yield f"I'm having a little trouble right now. Let me think..."

    async def handle_user_message_with_audio(
        self,
        story_id: str,
        user_message: str,
        tier: "InputTier",
        classification: Any,
        story: Any,
        chapter_triggering: bool = False,
        chapter_number: Optional[int] = None
    ) -> tuple:
        """
        Handle user message and return BOTH text response AND audio bytes.

        Uses direct audio output (GPT-4o-audio-preview) when enabled for ~50% latency reduction.
        Falls back to Claude + TTS pipeline when direct audio is disabled.

        Args:
            story_id: The story ID
            user_message: The user's message
            tier: Input tier classification
            classification: Full classification object
            story: Story object from Firebase
            chapter_triggering: If True, chapter generation is about to start
            chapter_number: Optional chapter number being generated (for continuation)

        Returns:
            Tuple of (response_text: str, audio_bytes: bytes or None)
        """
        from src.config import get_settings
        settings = get_settings()

        # Build the prompt context (same as handle_user_message)
        rich_context = self._build_rich_context(story)
        story_language = story.preferences.language if story and story.preferences else "en"
        lang_instruction = get_dialogue_instruction(story_language)
        lang_style = get_language_style(story_language)

        # Build system prompt for direct audio
        system_prompt = f"""{self._persona_prompt}

{lang_instruction}

CRITICAL RULES:
1. You MUST reference specific details from the STORY CONTEXT provided below
2. Mention character names, plot points, or story themes in your response
3. Respond warmly and briefly (2-3 sentences) in {lang_style['name']}
4. Use simple words appropriate for children
5. NEVER give generic responses like "What would you like to know?"
6. ALWAYS connect your answer to the specific story being created"""

        # Build the user context message
        if chapter_triggering:
            chapter_str = f"Chapter {chapter_number}" if chapter_number else "the story"
            if chapter_number and chapter_number > 1:
                context_message = f"""The child wants to CONTINUE the story and Chapter {chapter_number} generation is STARTING NOW!

Child said: "{user_message}"

Generate a SHORT excited response (1-2 sentences) announcing Chapter {chapter_number} is being written.
Examples (translate to {lang_style['name']}):
- "Absolutt! La meg skrive kapittel {chapter_number} med en gang - det blir spennende!"
- "Absolutely! Let me write Chapter {chapter_number} right now - it's going to be exciting!"
- "Ja! Kapittel {chapter_number} er på vei - la oss se hva som skjer videre!"
"""
            else:
                context_message = f"""The child wants to start the story and Chapter 1 generation is STARTING NOW!

Child said: "{user_message}"

Generate a SHORT excited response (1-2 sentences) announcing the story is starting.
Examples:
- "Fantastisk! La meg begynne på eventyret ditt med en gang!"
- "Wonderful! Let me start your story right now!"
"""
        else:
            context_message = f"""=== STORY CONTEXT (USE THIS IN YOUR RESPONSE!) ===
{rich_context}
=== END CONTEXT ===

The child asked: "{user_message}"
Their intent: {getattr(classification, 'classified_intent', 'general question')}

IMPORTANT: Your response MUST reference specific details from the STORY CONTEXT above!
If they ask about characters, mention the character's name and traits.
If they ask about the story, reference the actual plot or theme.
Respond naturally in 2-3 sentences as Hanan in {lang_style['name']}. Be warm, specific, and GROUNDED in the story context."""

        # Try direct audio if enabled AND TTS is not disabled
        # In dev mode (disable_tts=True), skip audio generation entirely for faster response
        if settings.use_direct_audio and not settings.disable_tts:
            try:
                logger.info(f"🎵 Using direct audio path (GPT-4o-audio-preview)")
                text, audio_bytes, transcript = await voice_service.generate_dialogue_with_audio(
                    system_prompt=system_prompt,
                    user_message=context_message
                )
                logger.info(f"✅ Direct audio: {len(text)} chars text, {len(audio_bytes)} bytes audio")
                return text, audio_bytes
            except Exception as e:
                logger.warning(f"⚠️ Direct audio failed, falling back to Claude+TTS: {e}")
                # Fall through to traditional path
        elif settings.disable_tts:
            logger.info(f"🔇 TTS disabled - using fast text-only path (LLM Router)")
            # Skip audio entirely, use LLM Router for fast text response

        # Fallback: Use Claude for text, then TTS separately
        text = await self.handle_user_message(
            story_id=story_id,
            user_message=user_message,
            tier=tier,
            classification=classification,
            story=story,
            chapter_triggering=chapter_triggering,
            chapter_number=chapter_number
        )

        # Generate audio with TTS
        tts_language = get_tts_language_code(story_language)
        audio_bytes = await voice_service.text_to_speech(
            text=text,
            speaking_rate=0.9,
            language_code=tts_language,
            use_case="dialogue"
        )

        return text, audio_bytes

    async def _store_style_preference(self, story_id: str, user_message: str):
        """
        Store a user's style preference in Firebase for agents to use.

        This is called when TIER_2_PREFERENCE is detected (e.g., "make it funny").
        The preference is stored in story.preferences.user_style_requests and
        will be passed to StructureAgent and NarrativeAgent for ALL chapters.
        """
        try:
            success = await self.firebase.append_style_request(story_id, user_message)
            if success:
                logger.info(f"💾 Stored style preference for {story_id}: {user_message}")
            else:
                logger.warning(f"⚠️ Failed to store style preference for {story_id}")
        except Exception as e:
            logger.error(f"Error storing style preference: {e}")

    async def _handle_influence(
        self,
        story_id: str,
        user_message: str,
        tier: InputTier,
        classification: Any,
        rich_context: str,
        language: str = "en"
    ) -> str:
        """Handle Tier 2-4 influence requests with enthusiastic acknowledgment"""
        # Get language-specific dialogue instruction
        lang_instruction = get_dialogue_instruction(language)
        lang_style = get_language_style(language)

        direction = getattr(classification, 'story_direction', user_message)

        # TIER_2 (style preferences) needs different handling than TIER_3/4 (story ideas)
        if tier == InputTier.TIER_2_PREFERENCE:
            # Style preference - affects the whole story's tone, not a specific plot element
            prompt = f"""{self._persona_prompt}

{lang_instruction}

The child wants the story to have a certain style/tone: "{user_message}"

Story context: {rich_context[:300]}

Respond naturally in 2-3 flowing sentences as Hanan in {lang_style['name']}.
Show enthusiasm for their preference, confirm you'll make the WHOLE story feel that way,
and maybe connect it to existing characters.

IMPORTANT: This is a STYLE preference for the whole story. Never say "your idea will appear in Chapter X".
Write as natural speech, not a numbered list.

Example: "Så gøy! Jeg elsker at du vil ha det morsomt! La meg sørge for at Harald og vennene hans har masse morsomme øyeblikk gjennom hele eventyret!"

Return ONLY the dialogue text in {lang_style['name']}."""
        else:
            # TIER_3 or TIER_4 - actual story additions/ideas
            tier_name = {
                InputTier.TIER_3_STORY_CHOICE: "story idea",
                InputTier.TIER_4_ADDITION: "creative addition"
            }.get(tier, "idea")

            prompt = f"""{self._persona_prompt}

{lang_instruction}

The child wants to add a {tier_name}: "{user_message}"
Their idea: {direction}

Story context: {rich_context[:300]}

Respond naturally in 2-3 flowing sentences as Hanan in {lang_style['name']}.
Show genuine excitement, connect their idea to existing characters by name,
and confirm you'll weave it into the story.

Write as natural speech, not a numbered list.

Example: "Oi, en regnbue! For en nydelig idé! Jeg skal sørge for at verdenen vår har en - kanskje Harald ser den på reisen sin!"

Return ONLY the dialogue text in {lang_style['name']}."""

        try:
            # Use LLM Router to respect models.yaml configuration
            result = await self._call_with_llm_router(prompt)

            # Store TIER_2 preferences in story for agents to use
            if tier == InputTier.TIER_2_PREFERENCE:
                await self._store_style_preference(story_id, user_message)

            return result["content"]  # Extract content from result dict
        except Exception as e:
            logger.error(f"CompanionAgent influence response error: {e}")
            # Fallback messages by language - different for preferences vs ideas
            if tier == InputTier.TIER_2_PREFERENCE:
                # Still try to store the preference even on response error
                await self._store_style_preference(story_id, user_message)
                fallbacks = {
                    "no": "Så gøy! Jeg elsker den ideen! La oss gjøre hele historien sånn!",
                    "es": "¡Qué divertido! ¡Me encanta esa idea! ¡Hagamos toda la historia así!",
                    "en": "How fun! I love that idea! Let's make the whole story like that!"
                }
            else:
                fallbacks = {
                    "no": "For en fantastisk idé! Jeg elsker det! Jeg skal sørge for å legge det til i historien din!",
                    "es": "¡Qué idea tan maravillosa! ¡Me encanta! ¡Me aseguraré de añadirla a tu historia!",
                    "en": "What a wonderful idea! I love it! I'll make sure to add that to your story!"
                }
            return fallbacks.get(language, fallbacks["en"])

    # ===== Event Handlers =====

    async def _on_structure_ready(self, event: StoryEvent):
        """Handle structure_ready event - announce story setup"""
        story_id = event.story_id
        if story_id not in self._states:
            return

        state = self._states[story_id]
        if state.announced_structure or not state.active:
            return

        # === OBSERVATORY: Emit agent_started for proactive action ===
        await story_events.emit("agent_started", story_id, {
            "agent": "CompanionAgent",
            "task": "Announcing story structure"
        })

        data = event.data
        title = data.get("title", "your story")
        theme = data.get("theme", "")

        # Fetch story for language preference
        story = await self.firebase.get_story(story_id)
        story_language = story.preferences.language if story and story.preferences else "en"
        lang_instruction = get_dialogue_instruction(story_language)
        lang_style = get_language_style(story_language)

        prompt = f"""{self._persona_prompt}

{lang_instruction}

The story structure is ready! Announce this exciting milestone to the child.

Story title: "{title}"
Theme: {theme}

Write 2-3 natural, flowing sentences in {lang_style['name']}.
Share your excitement that the story is taking shape, mention the title,
and tease what's coming. Write as natural speech, not a numbered list.

Be warm and build anticipation. Return ONLY the dialogue text in {lang_style['name']}."""

        try:
            # Use LLM Router to respect models.yaml configuration
            result = await self._call_with_llm_router(prompt)
            actual_model = result.get("model", "unknown")

            await self._send_proactive_message(story_id, result["content"], {
                "event": "structure_ready",
                "title": title
            }, story_language)

            state.announced_structure = True
            logger.info(f"CompanionAgent announced structure for {story_id} ({lang_style['name']})")

            # === OBSERVATORY: Emit agent_completed with actual model ===
            await story_events.emit("agent_completed", story_id, {
                "agent": "CompanionAgent",
                "success": True,
                "model": actual_model
            })

        except Exception as e:
            logger.error(f"CompanionAgent structure announcement error: {e}")
            # === OBSERVATORY: Emit agent_completed on error ===
            await story_events.emit("agent_completed", story_id, {
                "agent": "CompanionAgent",
                "success": False
            })

    async def _on_character_ready(self, event: StoryEvent):
        """Handle character_ready event - spotlight the new character"""
        story_id = event.story_id
        if story_id not in self._states:
            return

        state = self._states[story_id]
        if not state.active:
            return

        data = event.data
        char_name = data.get("name", "")

        # Skip if already announced this character
        if char_name in state.announced_characters:
            return

        char_role = data.get("role", "character")
        char_traits = data.get("personality_traits", [])
        char_motivation = data.get("motivation", "")
        char_background = data.get("background", "")

        # Fetch story for language preference
        story = await self.firebase.get_story(story_id)
        story_language = story.preferences.language if story and story.preferences else "en"
        lang_instruction = get_dialogue_instruction(story_language)
        lang_style = get_language_style(story_language)

        # Use quality model for character spotlights if available
        prompt = f"""{self._persona_prompt}

{lang_instruction}

A new character was just created! Introduce them to the child with warmth and excitement.

Character: {char_name} ({char_role})
Personality: {', '.join(char_traits[:3]) if char_traits else 'brave and curious'}
Background: {char_background[:150] if char_background else 'A mysterious past...'}

Write 2-3 natural, flowing sentences as Hanan in {lang_style['name']}.
Introduce the character with excitement, share something interesting about them,
and tease their role in the story. Write as natural speech, not a numbered list.

IMPORTANT: Do NOT always end with a question like "vil du treffe ham?"
Vary your endings naturally:
- Sometimes an excited statement ("I can't wait to see what they do!")
- Sometimes a teasing observation ("Something tells me they'll surprise us...")
- Occasionally a question, but not every time

Return ONLY the dialogue text in {lang_style['name']}."""

        try:
            # === OBSERVATORY: Emit agent_started for character spotlight ===
            await story_events.emit("agent_started", story_id, {
                "agent": "CompanionAgent",
                "task": f"Spotlighting {char_name}"
            })

            # Use LLM Router to respect models.yaml configuration
            result = await self._call_with_llm_router(prompt)
            actual_model = result.get("model", "unknown")

            await self._send_proactive_message(story_id, result["content"], {
                "event": "character_spotlight",
                "character": char_name,
                "role": char_role
            }, story_language)

            state.announced_characters.add(char_name)
            logger.info(f"CompanionAgent spotlighted character {char_name} for {story_id} ({lang_style['name']})")

            # === OBSERVATORY: Emit agent_completed with actual model ===
            await story_events.emit("agent_completed", story_id, {
                "agent": "CompanionAgent",
                "success": True,
                "model": actual_model
            })

        except Exception as e:
            logger.error(f"CompanionAgent character spotlight error: {e}")
            # === OBSERVATORY: Emit agent_completed on error ===
            await story_events.emit("agent_completed", story_id, {
                "agent": "CompanionAgent",
                "success": False
            })

    async def _on_chapter_ready(self, event: StoryEvent):
        """Handle chapter_ready event - AUTO-START playback (no confirmation needed)."""
        from src.services.events import story_events

        story_id = event.story_id
        if story_id not in self._states:
            return

        state = self._states[story_id]
        data = event.data
        chapter_num = data.get("chapter_number", 1)
        chapter_title = data.get("title", "")

        # Deactivate proactive teasers - chapter is ready!
        if chapter_num == 1:
            await self.deactivate(story_id)

        # Fetch story for language preference
        story = await self.firebase.get_story(story_id)
        story_language = story.preferences.language if story and story.preferences else "en"

        # Generate brief intro (no confirmation needed - just announce and play)
        intro_messages = {
            "no": f"Nå kommer kapittel {chapter_num}: {chapter_title}! Gjør deg klar!",
            "en": f"Here comes chapter {chapter_num}: {chapter_title}! Get ready!",
            "es": f"¡Aquí viene el capítulo {chapter_num}: {chapter_title}! ¡Prepárate!"
        }
        intro = intro_messages.get(story_language, intro_messages["en"])

        try:
            # === OBSERVATORY: Emit agent_started for chapter intro ===
            await story_events.emit("agent_started", story_id, {
                "agent": "CompanionAgent",
                "task": f"Introducing Chapter {chapter_num}"
            })

            # Send brief intro message
            await self._send_proactive_message(story_id, intro, {
                "event": "chapter_auto_start",
                "chapter": chapter_num,
                "title": chapter_title
            }, story_language)

            # Immediately emit chapter_audio_play (no waiting for confirmation)
            await story_events.emit("chapter_audio_play", story_id, {
                "chapter_number": chapter_num,
                "title": chapter_title,
                "auto_started": True
            })

            logger.info(f"CompanionAgent AUTO-STARTED Chapter {chapter_num} for {story_id}")

            # === OBSERVATORY: Emit agent_completed ===
            await story_events.emit("agent_completed", story_id, {
                "agent": "CompanionAgent",
                "success": True
            })

        except Exception as e:
            logger.error(f"CompanionAgent chapter auto-start error: {e}")
            # === OBSERVATORY: Emit agent_completed on error ===
            await story_events.emit("agent_completed", story_id, {
                "agent": "CompanionAgent",
                "success": False
            })

    # ===== Proactive Engagement =====

    async def _proactive_scheduler(self, story_id: str, initial_context: Optional[Dict]):
        """
        Schedule proactive teasers during Chapter 1 generation.

        Sends educational teasers every proactive_interval seconds
        until Chapter 1 is ready or max_teasers is reached.
        """
        if story_id not in self._states:
            return

        state = self._states[story_id]

        # Wait a bit before first teaser (let structure/characters load)
        await asyncio.sleep(self.proactive_interval)

        while state.active and state.proactive_teasers_sent < self.max_teasers:
            if state.chapter_1_ready:
                break

            try:
                # Get current story context
                story = await self.firebase.get_story(story_id)
                if not story:
                    break

                await self._send_proactive_teaser(story_id, story)
                state.proactive_teasers_sent += 1
                state.last_proactive_time = datetime.now()

            except Exception as e:
                logger.error(f"CompanionAgent proactive teaser error: {e}")

            # Wait for next interval
            await asyncio.sleep(self.proactive_interval)

        logger.info(f"CompanionAgent proactive scheduler ended for {story_id} "
                   f"(sent {state.proactive_teasers_sent} teasers)")

    async def _send_proactive_teaser(self, story_id: str, story):
        """Generate and send an educational teaser"""
        state = self._states.get(story_id)
        if not state or not state.active:
            return

        # Get language preference
        story_language = story.preferences.language if story and story.preferences else "en"
        lang_instruction = get_dialogue_instruction(story_language)
        lang_style = get_language_style(story_language)

        # Determine teaser type based on what's available
        teaser_type = self._choose_teaser_type(story, state)

        prompt = f"""{self._persona_prompt}

{lang_instruction}

While the story is being created, share something fun and educational with the child.

Story theme: {story.structure.theme if story.structure else 'adventure'}
Story setting: {story.prompt[:100] if story.prompt else 'a magical world'}

Topics already covered: {', '.join(state.educational_topics_covered) if state.educational_topics_covered else 'none yet'}

Write 2-3 sentences as Hanan in {lang_style['name']}:
- Type: {teaser_type}
- Share a "Did you know?" fact related to the story theme
- Make it age-appropriate and exciting
- End with something that builds anticipation

Examples (translate to {lang_style['name']}):
- "Did you know Vikings actually named their swords? I wonder what Harald will call his..."
- "Here's something cool - real castles had secret passages! Maybe there's one in our story..."

Return ONLY the dialogue text in {lang_style['name']}, nothing else."""

        try:
            # === OBSERVATORY: Emit agent_started for proactive teaser ===
            await story_events.emit("agent_started", story_id, {
                "agent": "CompanionAgent",
                "task": f"Sending teaser #{state.proactive_teasers_sent + 1}"
            })

            # Use LLM Router to respect models.yaml configuration
            result = await self._call_with_llm_router(prompt)
            actual_model = result.get("model", "unknown")

            await self._send_proactive_message(story_id, result["content"], {
                "event": "educational_teaser",
                "teaser_number": state.proactive_teasers_sent + 1
            }, story_language)

            # Track covered topics (simple extraction)
            if story.structure and story.structure.theme:
                state.educational_topics_covered.add(story.structure.theme)

            logger.info(f"CompanionAgent sent proactive teaser #{state.proactive_teasers_sent + 1} for {story_id} ({lang_style['name']})")

            # === OBSERVATORY: Emit agent_completed with actual model ===
            await story_events.emit("agent_completed", story_id, {
                "agent": "CompanionAgent",
                "success": True,
                "model": actual_model
            })

        except Exception as e:
            logger.error(f"CompanionAgent teaser generation error: {e}")
            # === OBSERVATORY: Emit agent_completed on error ===
            await story_events.emit("agent_completed", story_id, {
                "agent": "CompanionAgent",
                "success": False
            })

    def _choose_teaser_type(self, story, state: CompanionState) -> str:
        """Choose what type of teaser to send based on context"""
        teaser_types = [
            "educational_fact",
            "did_you_know",
            "story_anticipation",
            "character_tease"
        ]

        # Rotate through types
        idx = state.proactive_teasers_sent % len(teaser_types)
        return teaser_types[idx]

    # ===== Helper Methods =====

    def _is_affirmative_response(self, message: str) -> bool:
        """Check if message is an affirmative response (yes, ok, ready, etc.)"""
        message_lower = message.lower().strip()

        # Multi-language affirmative patterns
        affirmative_patterns = [
            # English
            r"^(yes|yeah|yep|yup|ok|okay|sure|ready|let'?s go|go|start|play)!*$",
            r"^(i'?m ready|let'?s do it|let'?s hear it|i want to hear)!*$",
            # Norwegian
            r"^(ja|jo|japp|jepp|ok|okei|greit|klar|kjør|start|spill)!*$",
            r"^(jeg er klar|la oss høre|jeg vil høre|kjør på)!*$",
            # Spanish
            r"^(sí|si|claro|vale|listo|vamos|empieza)!*$",
        ]

        import re
        for pattern in affirmative_patterns:
            if re.match(pattern, message_lower, re.IGNORECASE):
                return True
        return False

    async def _trigger_chapter_playback(self, story_id: str, story) -> str:
        """
        Trigger chapter playback after user confirms they're ready.

        This method:
        1. Introduces the chapter with Hanan's voice
        2. Emits a chapter_audio_play event for the frontend to play narrator audio
        3. Clears the pending playback state
        """
        from src.services.events import story_events

        state = self._states.get(story_id)
        if not state or state.pending_chapter_playback is None:
            return "Something went wrong, let me check on that chapter..."

        chapter_num = state.pending_chapter_playback
        chapter_title = state.pending_chapter_title or f"Chapter {chapter_num}"

        # Clear pending state
        state.pending_chapter_playback = None
        state.pending_chapter_title = None

        # Get language preference
        story_language = story.preferences.language if story and story.preferences else "en"
        lang_instruction = get_dialogue_instruction(story_language)
        lang_style = get_language_style(story_language)

        # Generate short intro from Hanan
        prompt = f"""{self._persona_prompt}

{lang_instruction}

The child said they're ready to hear Chapter {chapter_num}: "{chapter_title}"!

Write a VERY SHORT handoff (1 sentence max) as Hanan in {lang_style['name']}:
- Quick excited acknowledgment like "Here we go!" or "Get comfy!"
- This is the LAST thing before the narrator starts reading

Keep it under 10 words. Return ONLY the text in {lang_style['name']}."""

        try:
            # Use LLM Router to respect models.yaml configuration
            result = await self._call_with_llm_router(prompt)

            # Emit event for frontend to play chapter audio
            # The narrator voice will take over from here
            await story_events.emit("chapter_audio_play", story_id, {
                "chapter_number": chapter_num,
                "title": chapter_title,
                "intro_complete": True
            })

            logger.info(f"CompanionAgent triggered chapter {chapter_num} playback for {story_id}")

            return result["content"]  # Extract content from result dict

        except Exception as e:
            logger.error(f"CompanionAgent chapter playback trigger error: {e}")
            return "La oss høre historien!" if story_language == "no" else "Let's hear the story!"

    def _build_rich_context(self, story) -> str:
        """Build compact context from story data for dialogue grounding.

        Optimized for fast responses - keeps context under ~500 tokens.
        """
        if not story:
            return "Story context not available yet."

        context_parts = []

        # Story basics (compact)
        if story.prompt:
            prompt_short = story.prompt[:100] + "..." if len(story.prompt) > 100 else story.prompt
            context_parts.append(f"Story: {prompt_short}")

        # Structure context (essential only)
        if story.structure:
            context_parts.append(f"Title: {story.structure.title}")
            if story.structure.theme:
                context_parts.append(f"Theme: {story.structure.theme}")

            # Chapter synopses (first 3, with 150-char limit)
            if story.structure.chapters:
                for ch in story.structure.chapters[:3]:
                    synopsis = ch.synopsis[:150] + "..." if ch.synopsis and len(ch.synopsis) > 150 else (ch.synopsis or "")
                    if synopsis:
                        context_parts.append(f"Ch{ch.number} '{ch.title}': {synopsis}")

        # Characters (top 6, with relationships for grounded responses)
        if story.characters:
            char_sections = []
            for char in story.characters[:6]:
                traits = ", ".join(char.personality_traits[:3]) if char.personality_traits else ""
                char_info = f"{char.name} ({char.role}): {traits}"

                # Include relationships - critical for answering "does X have friends?"
                if hasattr(char, 'relationships') and char.relationships:
                    rel_list = []
                    for rel_name, rel_desc in list(char.relationships.items())[:3]:
                        # Compact relationship description
                        rel_short = rel_desc[:80] + "..." if len(rel_desc) > 80 else rel_desc
                        rel_list.append(f"{rel_name}: {rel_short}")
                    if rel_list:
                        char_info += f"\n  Relationships: {'; '.join(rel_list)}"

                # Include brief background for context
                if hasattr(char, 'background') and char.background:
                    bg_short = char.background[:150] + "..." if len(char.background) > 150 else char.background
                    char_info += f"\n  Background: {bg_short}"

                char_sections.append(char_info)

            if char_sections:
                context_parts.append("Characters:\n" + "\n".join(char_sections))

        result = "\n".join(context_parts)

        # Log the context being built for transparency
        context_summary = result[:300].replace('\n', ' | ') if len(result) > 300 else result.replace('\n', ' | ')
        logger.info(f"📚 CompanionAgent context built ({len(result)} chars): {context_summary}...")

        return result

    async def _build_playback_context(self, story_id: str, chapter_number: int) -> PlaybackContext:
        """
        Build rich playback context for grounded post-chapter discussion.

        Unlike _build_rich_context() which is optimized for fast responses,
        this fetches FULL chapter content and ALL characters for quality
        discussion after chapter playback.

        Args:
            story_id: The story ID
            chapter_number: Current chapter number being played/discussed

        Returns:
            PlaybackContext with full story data for rich responses
        """
        # Fetch story from Firebase
        story = await self.firebase.get_story(story_id)
        if not story:
            return PlaybackContext(
                story_id=story_id,
                current_chapter=chapter_number,
                playback_phase="unknown"
            )

        # Get reading state for phase info
        reading_state = await self.firebase.get_reading_state(story_id)
        playback_phase = "unknown"
        if reading_state and hasattr(reading_state, 'playback_phase'):
            playback_phase = reading_state.playback_phase.value if hasattr(reading_state.playback_phase, 'value') else str(reading_state.playback_phase)

        # Get current chapter content (FULL, not truncated)
        current_chapter_content = None
        current_chapter_title = None
        vocabulary_words = []
        educational_points = []

        if story.chapters:
            for ch in story.chapters:
                if ch.chapter_number == chapter_number:
                    current_chapter_content = ch.content  # FULL content
                    current_chapter_title = ch.title
                    # Educational elements
                    if hasattr(ch, 'vocabulary_words') and ch.vocabulary_words:
                        vocabulary_words = ch.vocabulary_words
                    if hasattr(ch, 'educational_points') and ch.educational_points:
                        educational_points = ch.educational_points
                    break

        # Get ALL characters (not limited to 4)
        all_characters = []
        characters_featured = []
        if story.characters:
            for char in story.characters:
                char_dict = {
                    "name": char.name,
                    "role": char.role,
                    "personality_traits": char.personality_traits or [],
                    "backstory": char.backstory or "",
                    "arc": char.arc or ""
                }
                all_characters.append(char_dict)

        # Extract characters mentioned in current chapter
        if current_chapter_content and story.characters:
            for char in story.characters:
                if char.name.lower() in current_chapter_content.lower():
                    characters_featured.append(char.name)

        # Get previous chapter synopsis (if not first chapter)
        previous_chapter_synopsis = None
        if chapter_number > 1 and story.chapters:
            for ch in story.chapters:
                if ch.chapter_number == chapter_number - 1:
                    previous_chapter_synopsis = ch.synopsis or ch.content[:200] + "..." if ch.content else None
                    break

        # Chapter progress
        chapters_completed = chapter_number  # Current chapter is completing
        total_chapters = len(story.chapters) if story.chapters else 0

        # Child age derived from story difficulty level (no child_profile in Story model)
        child_age = 10  # Default
        if story.preferences:
            difficulty = getattr(story.preferences, 'difficulty', None)
            difficulty_str = str(difficulty.value if hasattr(difficulty, 'value') else difficulty).lower()
            if difficulty_str == 'easy':
                child_age = 6
            elif difficulty_str == 'hard':
                child_age = 12
        child_name = None  # Not stored in Story model

        return PlaybackContext(
            story_id=story_id,
            current_chapter=chapter_number,
            playback_phase=playback_phase,
            current_chapter_content=current_chapter_content,
            current_chapter_title=current_chapter_title,
            vocabulary_words=vocabulary_words,
            educational_points=educational_points,
            characters_featured=characters_featured,
            all_characters=all_characters,
            previous_chapter_synopsis=previous_chapter_synopsis,
            chapters_completed=chapters_completed,
            total_chapters=total_chapters,
            child_age=child_age,
            child_name=child_name
        )

    async def _send_proactive_message(self, story_id: str, message: str, metadata: Dict, language: str = "en"):
        """Send a proactive message with audio generation"""
        # Check if still active
        state = self._states.get(story_id)
        if not state:
            return

        # Get TTS language code
        tts_language = get_tts_language_code(language)

        # Generate audio with appropriate language (dialogue = proactive message)
        audio_bytes = await voice_service.text_to_speech(
            text=message,
            speaking_rate=0.9,
            language_code=tts_language,
            use_case="dialogue"
        )

        audio_base64 = None
        if audio_bytes:
            audio_base64 = voice_service.encode_audio_base64(audio_bytes)

        # Emit dialogue event
        await story_events.emit(
            "dialogue_ready",
            story_id,
            {
                "message": message,
                "audio": audio_base64,
                "source": "companion_agent",
                **metadata
            }
        )

        logger.debug(f"CompanionAgent sent proactive message for {story_id} ({language}): {message[:50]}...")

    # ===== Post-Chapter Discussion =====

    async def generate_post_chapter_discussion(
        self,
        story_id: str,
        chapter_number: int
    ) -> Dict[str, Any]:
        """
        Generate a post-chapter discussion prompt after chapter playback ends.

        Like a teacher finishing reading aloud, this method creates a warm,
        engaging discussion that references specific events from the chapter.

        Args:
            story_id: The story ID
            chapter_number: The chapter that just finished

        Returns:
            Dict with:
                - message: The discussion prompt text
                - audio: Base64-encoded audio (if TTS enabled)
                - discussion_type: "post_chapter"
                - chapter_number: The completed chapter
                - has_more_chapters: Whether there are more chapters
        """
        # Fetch story for language preference
        story = await self.firebase.get_story(story_id)
        story_language = story.preferences.language if story and story.preferences else "en"
        lang_instruction = get_dialogue_instruction(story_language)
        lang_style = get_language_style(story_language)
        tts_language = get_tts_language_code(story_language)

        # Build rich context for quality response
        context = await self._build_playback_context(story_id, chapter_number)

        # Load the companion prompt for persona guidance
        prompt_path = Path(__file__).parent.parent / "prompts" / "companion_agent.txt"
        persona_prompt = ""
        if prompt_path.exists():
            persona_prompt = prompt_path.read_text()

        # Build the discussion generation prompt
        chapter_info = f"Chapter {chapter_number}"
        if context.current_chapter_title:
            chapter_info = f"'{context.current_chapter_title}' (Chapter {chapter_number})"

        # Prepare character info
        characters_in_chapter = ", ".join(context.characters_featured) if context.characters_featured else "the main characters"

        # Prepare educational elements (if any)
        educational_notes = ""
        if context.vocabulary_words:
            words = [w.get('word', '') for w in context.vocabulary_words[:2] if w.get('word')]
            if words:
                educational_notes += f"\nVocabulary words featured: {', '.join(words)}"
        if context.educational_points:
            educational_notes += f"\nEducational elements: {', '.join(context.educational_points[:2])}"

        system_prompt = f"""{persona_prompt}

{lang_instruction}

=== POST-CHAPTER DISCUSSION ===

You just finished reading {chapter_info} to the child. Now engage them in a brief, warm discussion.
IMPORTANT: Respond in {lang_style['name']}.

CHAPTER CONTENT (reference this!):
{context.current_chapter_content[:2000] if context.current_chapter_content else "Chapter content not available"}
{educational_notes}

CHARACTERS IN THIS CHAPTER: {characters_in_chapter}

DISCUSSION GUIDELINES:
1. ALWAYS react to something specific that happened in the chapter
2. VARY your endings - sometimes a question, sometimes a statement or observation
3. Do NOT always end with a question - it becomes repetitive
4. Never make the child feel tested
5. Keep it to 2-3 sentences max
6. Sound genuinely excited about what just happened
7. If there's a cliffhanger, acknowledge it!
8. Respond in {lang_style['name']}

CHILD INFO:
- Age: {context.child_age or 'Unknown'}
- Name: {context.child_name or 'Young reader'}

GOOD EXAMPLES (translate to {lang_style['name']} - notice varied endings):
- "Wow, that was intense when [specific thing happened]! What do you think [character] will do now?"
- "I love how [character] was so [trait] in this chapter! That moment really touched me..."
- "Oh my goodness, I did NOT see that coming! I have a feeling the next chapter will be amazing!"
- "That scene with [character] was so powerful. I can still feel it..."
- "[Character] is really growing! I can't wait to see what happens next."

BAD EXAMPLES (avoid):
- "What did you think?" (too generic)
- "Can you tell me what happened?" (feels like a test)
- "Did you like it?" (yes/no question)
- Ending EVERY message with a question (repetitive)

Generate ONE warm, engaging post-chapter response in {lang_style['name']}."""

        try:
            # Use quality routing mode for rich discussion
            response = await self._foundry_service.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate the post-chapter discussion prompt."}
                ],
                routing_mode="quality",  # Quality mode for rich discussion
                max_tokens=1500,  # Reasoning models use 500-1000 internal tokens before output
                temperature=0.7
            )
            message = response["content"].strip()
            logger.debug(f"Post-chapter discussion used model: {response.get('model', 'unknown')}")
        except Exception as e:
            logger.error(f"Error generating post-chapter discussion: {e}")
            # Fallback messages by language
            fallbacks = {
                "no": f"For et spennende kapittel! Hva synes du om det {characters_in_chapter} gjorde?",
                "es": f"¡Qué capítulo tan emocionante! ¿Qué te pareció lo que hizo {characters_in_chapter}?",
                "en": f"That was such an exciting chapter! What did you think about what {characters_in_chapter} did?"
            }
            message = fallbacks.get(story_language, fallbacks["en"])

        # Generate audio with language (dialogue = post-chapter discussion)
        audio_base64 = None
        try:
            audio_bytes = await voice_service.text_to_speech(
                text=message,
                speaking_rate=0.9,
                language_code=tts_language,
                use_case="dialogue"
            )
            if audio_bytes:
                audio_base64 = voice_service.encode_audio_base64(audio_bytes)
        except Exception as e:
            logger.warning(f"TTS failed for post-chapter discussion: {e}")

        # Check if there are more chapters
        has_more_chapters = context.current_chapter < context.total_chapters

        return {
            "message": message,
            "audio": audio_base64,
            "discussion_type": "post_chapter",
            "chapter_number": chapter_number,
            "has_more_chapters": has_more_chapters,
            "source": "companion_agent"
        }

    async def handle_discussion_response(
        self,
        story_id: str,
        chapter_number: int,
        user_message: str,
        exchange_count: int = 1
    ) -> Dict[str, Any]:
        """
        Handle child's response during post-chapter discussion.

        Continues the discussion for 1-2 exchanges if the child is engaged,
        or gracefully transitions to next chapter if they signal readiness.

        Args:
            story_id: The story ID
            chapter_number: The chapter being discussed
            user_message: The child's response
            exchange_count: How many exchanges we've had (1-2 max)

        Returns:
            Dict with:
                - message: The response text
                - audio: Base64-encoded audio
                - continue_discussion: Whether to continue discussing
                - ready_for_next: Whether child wants to continue to next chapter
        """
        # Fetch story for language preference
        story = await self.firebase.get_story(story_id)
        story_language = story.preferences.language if story and story.preferences else "en"
        lang_instruction = get_dialogue_instruction(story_language)
        lang_style = get_language_style(story_language)
        tts_language = get_tts_language_code(story_language)

        # Check for "skip" signals - child wants to move on
        skip_signals = [
            "next", "continue", "keep going", "more", "what happens",
            "next chapter", "let's go", "read more", "keep reading",
            # Norwegian skip signals
            "neste", "fortsett", "mer", "hva skjer", "neste kapittel",
            # Spanish skip signals
            "siguiente", "continúa", "más", "qué pasa", "próximo capítulo"
        ]
        wants_to_continue = any(signal in user_message.lower() for signal in skip_signals)

        if wants_to_continue or exchange_count >= 2:
            # Transition to next chapter
            return await self._generate_chapter_transition(story_id, chapter_number)

        # Build context for continuing discussion
        context = await self._build_playback_context(story_id, chapter_number)

        # Load persona prompt
        prompt_path = Path(__file__).parent.parent / "prompts" / "companion_agent.txt"
        persona_prompt = ""
        if prompt_path.exists():
            persona_prompt = prompt_path.read_text()

        system_prompt = f"""{persona_prompt}

{lang_instruction}

=== CONTINUING DISCUSSION ===

You're discussing {context.current_chapter_title or f"Chapter {chapter_number}"} with the child.
They just said: "{user_message}"
IMPORTANT: Respond in {lang_style['name']}.

CHAPTER CONTEXT:
{context.current_chapter_content[:1500] if context.current_chapter_content else "Chapter content not available"}

GUIDELINES:
1. Respond warmly to what they said
2. Connect their response to something in the story
3. End with ONE of these (vary your approach!):
   - An engaging question about the story
   - An offer to continue the story
   - An excited observation or statement
4. Keep it to 2-3 sentences
5. If they seem eager for more story, offer to continue
6. Respond in {lang_style['name']}

This is exchange #{exchange_count + 1}. Prefer offering to continue the story.

CHILD INFO:
- Age: {context.child_age or 'Unknown'}
- Name: {context.child_name or 'Young reader'}"""

        try:
            # Use quality routing mode for rich discussion
            response = await self._foundry_service.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Child says: {user_message}"}
                ],
                routing_mode="quality",  # Quality mode for rich discussion
                max_tokens=1500,  # Reasoning models use 500-1000 internal tokens before output
                temperature=0.7
            )
            message = response["content"].strip()
            logger.debug(f"Discussion response used model: {response.get('model', 'unknown')}")
        except Exception as e:
            logger.error(f"Error generating discussion response: {e}")
            # Fallback messages by language
            fallbacks = {
                "no": "For en flott tanke! Er du klar til å finne ut hva som skjer videre?",
                "es": "¡Qué pensamiento tan genial! ¿Estás listo para descubrir qué pasa después?",
                "en": "That's such a great thought! Are you ready to find out what happens next?"
            }
            message = fallbacks.get(story_language, fallbacks["en"])

        # Generate audio with language (dialogue = discussion response)
        audio_base64 = None
        try:
            audio_bytes = await voice_service.text_to_speech(
                text=message,
                speaking_rate=0.9,
                language_code=tts_language,
                use_case="dialogue"
            )
            if audio_bytes:
                audio_base64 = voice_service.encode_audio_base64(audio_bytes)
        except Exception as e:
            logger.warning(f"TTS failed for discussion response: {e}")

        return {
            "message": message,
            "audio": audio_base64,
            "continue_discussion": exchange_count < 2,
            "ready_for_next": False,
            "source": "companion_agent"
        }

    async def _generate_chapter_transition(
        self,
        story_id: str,
        completed_chapter: int
    ) -> Dict[str, Any]:
        """Generate a warm transition to the next chapter."""
        # Fetch story for language preference
        story = await self.firebase.get_story(story_id)
        story_language = story.preferences.language if story and story.preferences else "en"
        tts_language = get_tts_language_code(story_language)

        context = await self._build_playback_context(story_id, completed_chapter)

        next_chapter = completed_chapter + 1
        has_next = next_chapter <= context.total_chapters

        # Transition messages by language
        if has_next:
            messages = {
                "no": f"Greit, la oss finne ut hva som skjer videre! Kapittel {next_chapter} er klart - her kommer det!",
                "es": f"¡Muy bien, vamos a descubrir qué pasa después! El capítulo {next_chapter} está listo - ¡aquí vamos!",
                "en": f"Alright, let's find out what happens next! Chapter {next_chapter} is ready - here we go!"
            }
        else:
            messages = {
                "no": "For en utrolig historie! Du har nådd slutten. Jeg håper du likte eventyret vårt sammen!",
                "es": "¡Qué historia tan increíble! Has llegado al final. ¡Espero que hayas disfrutado nuestra aventura juntos!",
                "en": "What an incredible story! You've reached the end. I hope you enjoyed our adventure together!"
            }
        message = messages.get(story_language, messages["en"])

        # Generate audio with language (dialogue = chapter transition)
        audio_base64 = None
        try:
            audio_bytes = await voice_service.text_to_speech(
                text=message,
                speaking_rate=0.9,
                language_code=tts_language,
                use_case="dialogue"
            )
            if audio_bytes:
                audio_base64 = voice_service.encode_audio_base64(audio_bytes)
        except Exception as e:
            logger.warning(f"TTS failed for chapter transition: {e}")

        return {
            "message": message,
            "audio": audio_base64,
            "continue_discussion": False,
            "ready_for_next": has_next,
            "next_chapter": next_chapter if has_next else None,
            "source": "companion_agent"
        }

    # ===== One-Chapter-At-A-Time Prefetch =====

    async def on_chapter_playback_start(self, story_id: str, chapter_number: int):
        """
        Called when user starts playing a chapter.
        Triggers prefetch of next chapter in background (non-blocking).

        For Chapter 1, also triggers Structure V2 refinement which rewrites
        synopses for chapters 2-N using D&D character cards and user preferences.

        Args:
            story_id: The story ID
            chapter_number: The chapter number that started playing
        """
        state = self._states.get(story_id)
        if not state:
            # Create minimal state if doesn't exist
            state = CompanionState(story_id=story_id)
            self._states[story_id] = state

        # === STRUCTURE V2: Trigger refinement when Chapter 1 starts ===
        # This runs in background to refine synopses for chapters 2-N
        # using full D&D character cards and user preferences
        if chapter_number == 1:
            logger.info(f"📚 Chapter 1 playback started - triggering Structure V2 refinement for {story_id}")
            asyncio.create_task(
                self._run_structure_v2_refinement(story_id)
            )

        next_chapter = chapter_number + 1

        # Check if already prefetching or prefetched
        if next_chapter in state.prefetching_chapters:
            logger.info(f"Chapter {next_chapter} already being prefetched for {story_id}")
            return
        if next_chapter in state.prefetched_chapters:
            logger.info(f"Chapter {next_chapter} already prefetched for {story_id}")
            return

        # Check if next chapter exists in structure
        if not await self._chapter_exists_in_structure(story_id, next_chapter):
            logger.info(f"Chapter {next_chapter} does not exist in structure for {story_id}")
            return

        # Check if chapter already written
        if await self._chapter_already_written(story_id, next_chapter):
            logger.info(f"Chapter {next_chapter} already written for {story_id}")
            state.prefetched_chapters.add(next_chapter)
            return

        # Mark as prefetching
        state.prefetching_chapters.add(next_chapter)

        # Emit prefetch event for routes.py to handle
        await story_events.emit(
            "prefetch_chapter",
            story_id,
            {
                "chapter_number": next_chapter,
                "triggered_by_chapter": chapter_number
            }
        )

        logger.info(f"CompanionAgent triggered prefetch of Chapter {next_chapter} for {story_id}")

    async def on_chapter_prefetch_complete(self, story_id: str, chapter_number: int):
        """
        Called when a chapter prefetch is complete.

        Args:
            story_id: The story ID
            chapter_number: The chapter that was prefetched
        """
        state = self._states.get(story_id)
        if not state:
            return

        state.prefetching_chapters.discard(chapter_number)
        state.prefetched_chapters.add(chapter_number)
        logger.info(f"Chapter {chapter_number} prefetch complete for {story_id}")

    async def _run_structure_v2_refinement(self, story_id: str):
        """
        Run Structure V2 refinement in background.

        This refines chapter synopses for chapters 2-N using:
        - Full D&D character cards with skills
        - User preferences from dialogue phase
        - Actual Chapter 1 content

        Non-blocking, non-fatal if it fails.
        """
        try:
            from src.api.routes import get_coordinator
            coordinator = get_coordinator()

            if not coordinator:
                logger.warning(f"Structure V2: No coordinator available for {story_id}")
                return

            result = await coordinator.refine_structure_v2(story_id)

            if result.get("success"):
                refined_count = result.get("refined_chapters", 0)
                skills = result.get("skills_leveraged", [])
                model_used = result.get("model", "unknown")
                logger.info(f"✅ Structure V2 complete for {story_id}: refined {refined_count} chapters, leveraged {len(skills)} skills (model: {model_used})")
            else:
                error = result.get("error", "Unknown error")
                if result.get("original_preserved"):
                    logger.warning(f"⚠️ Structure V2 failed for {story_id}: {error} (using original structure)")
                else:
                    logger.error(f"❌ Structure V2 failed for {story_id}: {error}")

        except Exception as e:
            logger.error(f"Structure V2 exception for {story_id}: {e}")

    async def _chapter_exists_in_structure(self, story_id: str, chapter_number: int) -> bool:
        """Check if a chapter exists in the story structure"""
        try:
            story = await self.firebase.get_story(story_id)
            if not story or not story.structure or not story.structure.chapters:
                return False

            # Check if chapter_number exists in the structure
            for ch in story.structure.chapters:
                if ch.number == chapter_number:
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking chapter structure: {e}")
            return False

    async def _chapter_already_written(self, story_id: str, chapter_number: int) -> bool:
        """Check if a chapter has already been written (has content)"""
        try:
            story = await self.firebase.get_story(story_id)
            if not story or not story.chapters:
                return False

            # Check if chapter_number has content
            for ch in story.chapters:
                if ch.number == chapter_number and ch.content:
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking chapter content: {e}")
            return False

    # ===== Chapter Question Detection and Generation Trigger =====

    def _detect_chapter_question(self, message: str) -> Optional[int]:
        """
        Detect if user is asking about a specific chapter number.

        Returns the chapter number if detected, None otherwise.
        """
        import re
        message_lower = message.lower()

        # Patterns for detecting chapter-specific questions
        patterns = [
            # English patterns
            r"(?:what happens|tell me about|read|play|start).*chapter\s*(\d+)",
            r"chapter\s*(\d+).*(?:what|tell|about|happens)",
            r"(?:go to|skip to|jump to).*chapter\s*(\d+)",
            # Norwegian patterns
            r"(?:hva skjer|fortell om|les|spill|start).*kapittel\s*(\d+)",
            r"kapittel\s*(\d+).*(?:hva|fortell|om|skjer)",
            r"(?:gå til|hopp til).*kapittel\s*(\d+)",
            # Spanish patterns
            r"(?:qué pasa|cuéntame|leer|empezar).*capítulo\s*(\d+)",
            r"capítulo\s*(\d+).*(?:qué|cuenta|pasa)",
        ]

        for pattern in patterns:
            match = re.search(pattern, message_lower)
            if match:
                return int(match.group(1))

        return None

    async def _handle_chapter_question(
        self,
        story_id: str,
        chapter_number: int,
        language: str = "en"
    ) -> Optional[str]:
        """
        Handle a question about a specific chapter.

        If the chapter isn't generated yet, triggers generation and returns
        an appropriate response. Returns None if chapter exists and should
        be handled normally.
        """
        # Check if chapter exists in structure
        if not await self._chapter_exists_in_structure(story_id, chapter_number):
            messages = {
                "no": f"Historien har ikke så mange kapitler ennå! La oss fokusere på det vi har.",
                "es": f"¡La historia no tiene tantos capítulos todavía! Enfoquémonos en lo que tenemos.",
                "en": f"The story doesn't have that many chapters yet! Let's focus on what we have."
            }
            return messages.get(language, messages["en"])

        # Check if chapter is already written
        if await self._chapter_already_written(story_id, chapter_number):
            # Chapter exists, let normal handling show it
            return None

        # Chapter not written yet - trigger generation!
        logger.info(f"📚 User asked about ungenerated Chapter {chapter_number} - triggering generation")

        # Trigger the prefetch event
        await story_events.emit(
            "prefetch_chapter",
            story_id,
            {
                "chapter_number": chapter_number,
                "triggered_by_chapter": chapter_number - 1
            }
        )

        # Mark as prefetching
        state = self._states.get(story_id)
        if state:
            state.prefetching_chapters.add(chapter_number)

        # Return a response about generating the chapter
        messages = {
            "no": f"Åh, du vil vite hva som skjer i kapittel {chapter_number}! Gi meg et lite øyeblikk mens jeg gjør det klart for deg...",
            "es": f"¡Oh, quieres saber qué pasa en el capítulo {chapter_number}! Dame un momentito mientras lo preparo para ti...",
            "en": f"Oh, you want to know what happens in chapter {chapter_number}! Give me just a moment while I prepare it for you..."
        }
        return messages.get(language, messages["en"])

    # ===== End-of-Book Comprehensive Question Handling =====

    async def handle_end_of_book_question(
        self,
        story_id: str,
        question: str,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Answer questions about a completed story with full comprehension.

        Uses PaginatedStoryContext to load only the context pages needed for
        the specific question, balancing comprehensiveness with token efficiency.

        Args:
            story_id: Story ID
            question: Child's question about the story
            language: Language code for response

        Returns:
            Dict with response text, audio (if available), and context used
        """
        logger.info(f"End-of-book question for {story_id}: {question[:50]}...")

        # Create paginated context loader
        context = PaginatedStoryContext(story_id=story_id, firebase=self.firebase)

        # Determine which pages are needed based on the question
        pages_needed = self._classify_question_context_needs(question)
        logger.info(f"Loading context pages: {pages_needed}")

        # Load only required pages
        context_parts = []
        for page in pages_needed:
            page_data = await context.get_page(page)
            if page_data and "error" not in page_data:
                context_parts.append(f"=== {page.upper()} ===\n{self._format_context_page(page, page_data)}")

        # Get language instruction
        language_style = get_language_style(language)
        language_instruction = get_dialogue_instruction(language)

        # Build prompt for comprehensive answer
        prompt = f"""{self._persona_prompt}

{language_instruction}

The story has ENDED. The child is asking about the complete story they just heard.

STORY CONTEXT (loaded on-demand for this question):
{chr(10).join(context_parts)}

CHILD'S QUESTION: "{question}"

Instructions:
1. Answer comprehensively, referencing SPECIFIC story events, character names, and plot developments
2. If the question is about characters, mention how they grew and changed
3. If the question is about plot, reference specific chapters or events
4. Keep your response warm and engaging (2-4 sentences for simple questions, up to 6 for complex ones)
5. If you don't have enough context to answer, be honest but encouraging

Respond in {language_style.get('language_name', 'English')} as Hanan would."""

        try:
            if self._foundry_service:
                response = await self._foundry_service.chat(
                    messages=[{"role": "user", "content": prompt}],
                    routing_mode="quality"  # Use quality mode for comprehensive answers
                )
                response_text = response.get("content", "")
            else:
                response_text = "I'm sorry, I can't answer questions right now. Please try again!"

            # Generate audio if enabled
            audio_base64 = None
            tts_language = get_tts_language_code(language)
            try:
                audio_base64 = await voice_service.generate_dialogue_audio_async(
                    response_text, tts_language
                )
            except Exception as e:
                logger.warning(f"TTS failed for end-of-book response: {e}")

            return {
                "text": response_text,
                "audio": audio_base64,
                "context_pages_used": pages_needed,
                "source": "companion_agent_end_of_book"
            }

        except Exception as e:
            logger.error(f"End-of-book question failed: {e}")
            return {
                "text": "I'm having trouble remembering that part of the story. Can you ask me another question?",
                "audio": None,
                "error": str(e),
                "source": "companion_agent_end_of_book"
            }

    def _classify_question_context_needs(self, question: str) -> List[str]:
        """
        Determine which context pages are needed for this question.

        Analyzes the question to load only relevant context, reducing token usage
        while ensuring comprehensive answers.

        Args:
            question: The child's question

        Returns:
            List of ContextPage constants to load
        """
        q_lower = question.lower()
        pages = [ContextPage.OVERVIEW]  # Always include overview for context

        # Character-related keywords
        character_keywords = [
            "character", "person", "who", "friend", "hero", "villain",
            "name", "protagonist", "hovedperson", "karakter", "personaje",
            "how did", "what happened to", "why did"
        ]
        if any(word in q_lower for word in character_keywords):
            pages.append(ContextPage.CHARACTERS)

        # Plot/mystery-related keywords
        plot_keywords = [
            "what happened", "mystery", "secret", "letter", "object",
            "why", "how", "reveal", "discover", "find", "plot",
            "hemmelighet", "mysterium", "oppdage"
        ]
        if any(word in q_lower for word in plot_keywords):
            pages.append(ContextPage.PLOT_THREADS)

        # Story structure keywords
        chapter_keywords = [
            "chapter", "beginning", "end", "middle", "story", "favorite part",
            "kapittel", "slutt", "begynnelse", "historie"
        ]
        if any(word in q_lower for word in chapter_keywords):
            pages.append(ContextPage.CHAPTERS)

        # If question is very general or we're uncertain, load more context
        if len(pages) == 1:  # Only overview
            # Default to characters + chapters for general questions
            pages.extend([ContextPage.CHARACTERS, ContextPage.CHAPTERS])

        return pages

    def _format_context_page(self, page: str, data: Dict) -> str:
        """Format a context page for inclusion in the prompt."""
        import json

        if page == ContextPage.OVERVIEW:
            return f"Title: {data.get('title', 'Unknown')}\nTheme: {data.get('theme', 'Unknown')}\nChapters: {data.get('total_chapters', 0)}"

        elif page == ContextPage.CHARACTERS:
            chars = data.get("characters", [])
            if not chars:
                return "No characters available."
            lines = []
            for c in chars:
                lines.append(f"- {c['name']} ({c['role']}): {c['arc_summary']}")
            return "\n".join(lines)

        elif page == ContextPage.PLOT_THREADS:
            elements = data.get("plot_elements", [])
            if not elements:
                return "No plot threads tracked."
            lines = []
            for pe in elements:
                status = "✅ Resolved" if pe["status"] == "resolved" else "⏳ Pending"
                lines.append(f"- {pe['name']} ({pe['type']}): {status}")
            return "\n".join(lines)

        elif page == ContextPage.CHAPTERS:
            chapters = data.get("chapters", [])
            if not chapters:
                return "No chapters available."
            lines = []
            for ch in chapters:
                lines.append(f"Ch{ch['number']}: {ch['title']} - {ch['synopsis'][:100]}...")
            return "\n".join(lines)

        return json.dumps(data, indent=2)
