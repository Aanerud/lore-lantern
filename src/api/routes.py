"""
API routes for Lore Lantern

REST endpoints for story creation and management.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
import sys
from pathlib import Path
import asyncio
import logging
import uuid
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models import (
    StoryInitRequest,
    StoryInitResponse,
    StoryStatusResponse,
    StoryPreferences,
    StoryStatus,
    QueuedInput,
    InputTier,
    # New profile models
    ParentAccount,
    ParentAccountCreate,
    ParentAccountUpdate,
    ChildProfile,
    ChildProfileCreate,
    ChildProfileUpdate,
    ChildProfileResponse,
    LearningProgress,
    StoryLibraryItem,
    StoryLibraryResponse
)
from src.crew import StoryCrewCoordinator
from src.config import get_settings
from src.config.limits import PROMPT_MAX_LENGTH, MESSAGE_MAX_LENGTH
from src.utils.language_styles import get_tts_language_code
from src.services.events import story_events
from src.services.voice import voice_service
from src.agents.companion import get_companion_agent, DialogueResponse
from src.services.intent_classifier import classify_conversation_intent, ConversationIntent
import base64
from pydantic import BaseModel, ValidationError
from typing import Optional

# Logger for API routes
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["stories"])

# Global services (will be set by main app)
_coordinator: StoryCrewCoordinator = None
_blob_storage_service = None


def set_coordinator(coordinator: StoryCrewCoordinator):
    """Set the global coordinator instance"""
    global _coordinator
    _coordinator = coordinator


def set_blob_storage_service(service):
    """Set the global blob storage service instance (avoids circular import with main.py)"""
    global _blob_storage_service
    _blob_storage_service = service


def get_coordinator() -> StoryCrewCoordinator:
    """Get the global coordinator instance"""
    return _coordinator


async def _trigger_chapter_generation_from_routes(story_id: str, chapter_number: int):
    """
    Trigger chapter generation in background (called from routes.py).

    Used when user selects a story to continue from /conversation/continue
    and the story has structure but the chapter hasn't been written yet.
    """
    try:
        logger.info(f"📝 [routes] Background chapter {chapter_number} generation starting for {story_id}")

        if _coordinator is None:
            logger.error("Coordinator not initialized - cannot generate chapter")
            return

        # Generate the chapter
        result = await _coordinator.write_chapter(story_id, chapter_number)

        if result.get("success"):
            logger.info(f"✅ [routes] Chapter {chapter_number} generated successfully for {story_id}")
        else:
            logger.error(f"❌ [routes] Chapter generation failed: {result.get('error')}")

    except Exception as e:
        logger.error(f"❌ [routes] Error in background chapter generation: {e}", exc_info=True)


async def _trigger_audio_generation_from_routes(story_id: str, chapter_number: int):
    """
    Trigger audio generation for an existing chapter and emit via WebSocket.

    Used when user resumes a story that has chapter content but hasn't played audio yet.
    """
    try:
        logger.info(f"🔊 [routes] Background audio generation starting for {story_id} chapter {chapter_number}")

        if _coordinator is None:
            logger.error("Coordinator not initialized - cannot generate audio")
            return

        # Get story and chapter
        story = await _coordinator.storage.get_story(story_id)
        if not story:
            logger.error(f"Story {story_id} not found")
            return

        chapter = None
        for ch in story.chapters or []:
            if ch.number == chapter_number:
                chapter = ch
                break

        if not chapter or not chapter.content:
            logger.error(f"Chapter {chapter_number} not found or has no content")
            return

        # Get language
        lang_code = story.preferences.language if story.preferences else "en"
        language = get_tts_language_code(lang_code)

        # Generate audio (narration = story chapters)
        logger.info(f"🔊 Generating audio for chapter {chapter_number} ({len(chapter.content)} chars, lang: {language})")
        audio_bytes = await voice_service.text_to_speech(
            text=chapter.content,
            language_code=language,
            use_case="narration"
        )

        if not audio_bytes:
            logger.error("TTS generation failed - no audio returned")
            await story_events.emit("error", story_id, {
                "stage": "audio_generation",
                "chapter_number": chapter_number,
                "error": "Audio generation failed"
            })
            return

        # Emit audio via WebSocket
        import base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        audio_size_mb = len(audio_base64) / (1024 * 1024)

        logger.info(f"✅ [routes] Audio generated for chapter {chapter_number}: {len(audio_bytes)} bytes ({audio_size_mb:.1f}MB base64)")

        # Check if audio is too large for WebSocket (limit ~16MB typically)
        if audio_size_mb > 10:
            logger.warning(f"⚠️ Audio is large ({audio_size_mb:.1f}MB) - may fail WebSocket send")

        # Emit chapter_audio event for frontend to play
        logger.info(f"📡 Emitting chapter_audio event for {story_id}...")
        try:
            await story_events.emit("chapter_audio", story_id, {
                "chapter_number": chapter_number,
                "chapter_title": chapter.title if hasattr(chapter, 'title') else f"Chapter {chapter_number}",
                "audio_base64": audio_base64,
                "content_length": len(chapter.content),
                "auto_play": True
            })
            logger.info(f"✅ [routes] chapter_audio event emitted successfully")
        except Exception as emit_error:
            logger.error(f"❌ Failed to emit chapter_audio: {emit_error}", exc_info=True)

    except Exception as e:
        logger.error(f"❌ [routes] Error in background audio generation: {e}", exc_info=True)


# ===== Conversation-First Endpoint =====

class ConversationStartRequest(BaseModel):
    """Request body for starting a conversation (before story generation)."""
    message: str  # The child's initial message
    child_id: str  # Child profile ID
    language: Optional[str] = "en"  # Language preference (en, no, es)


class ConversationStartResponse(BaseModel):
    """Response from conversation start endpoint."""
    dialogue: str  # CompanionAgent's response text
    audio: Optional[str] = None  # Base64 encoded audio
    intent: str  # Detected intent: "continue", "new_story", "exploring", "greeting"
    suggested_action: str  # Frontend action: "resume_story", "init_story", "ask_preference", "greet"
    active_story_id: Optional[str] = None  # If child has an active story
    child_name: str  # For personalization
    child_age: int  # For age-appropriate responses


@router.post("/conversation/start", response_model=ConversationStartResponse)
async def start_conversation(request: ConversationStartRequest):
    """
    Start a conversation with CompanionAgent WITHOUT immediately starting story generation.

    This enables the "conversation-first" UX where:
    1. Child says something (anything)
    2. CompanionAgent engages first
    3. System detects intent (continue, new_story, exploring)
    4. Frontend decides next action based on response

    Use cases:
    - "I want to continue the story from last time" → detects continue intent
    - "Tell me a story about dragons" → detects new_story intent
    - "Hi!" or vague messages from young children → exploring mode

    Returns dialogue + intent so frontend can route appropriately.
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    companion = get_companion_agent()
    if not companion:
        raise HTTPException(status_code=500, detail="CompanionAgent not initialized")

    # Get child profile
    child = await _coordinator.storage.get_child_profile(request.child_id)
    if not child:
        raise HTTPException(status_code=404, detail=f"Child profile not found: {request.child_id}")

    # Calculate child's age
    current_year = datetime.now().year
    child_age = current_year - child.birth_year

    # Get parent's language preference if not specified
    language = request.language
    if not language or language == "en":
        parent = await _coordinator.storage.get_parent_account(child.parent_id)
        if parent and parent.language:
            language = parent.language

    # Get ALL child's stories (not just active)
    all_stories = await _coordinator.storage.get_stories_by_child(request.child_id)

    # Build story summaries for CompanionAgent context
    story_summaries = []
    active_story = None
    for story in all_stories:
        if story.id == child.active_story_id:
            active_story = story

        # Create summary for each story
        summary = {
            "story_id": story.id,
            "title": story.structure.title if story.structure else "Untitled",
            "theme": story.structure.theme if story.structure else None,
            "is_active": story.id == child.active_story_id,
            "chapters_completed": len([ch for ch in (story.chapters or []) if ch.status == "completed"]),
            "total_chapters": len(story.structure.chapters) if story.structure and story.structure.chapters else 0,
        }
        story_summaries.append(summary)

    logger.info(f"📚 Child {child.name} has {len(story_summaries)} stories")

    # Classify intent
    intent_result = await classify_conversation_intent(
        message=request.message,
        has_active_story=active_story is not None,
        child_age=child_age,
        language=language
    )

    logger.info(f"🎯 Intent classified: {intent_result.intent.value} (confidence: {intent_result.confidence})")

    # Emit conversation context event for the user's message (for Observatory)
    story_id_for_event = child.active_story_id or "pre_story"
    await story_events.emit(
        "conversation_context", story_id_for_event,
        {
            "speaker": "user",
            "name": child.name,
            "message": request.message,
            "intent": intent_result.intent.value,
            "confidence": intent_result.confidence
        }
    )

    # Determine suggested action - override if multiple stories need selection
    suggested_action = intent_result.suggested_action
    if intent_result.intent.value == "continue" and len(story_summaries) > 1:
        # Multiple stories - child needs to choose which one
        suggested_action = "ask_preference"
        logger.info(f"📚 Multiple stories ({len(story_summaries)}) - asking for preference")

    # Generate greeting response via CompanionAgent
    greeting = await companion.generate_greeting(
        child_name=child.name,
        message=request.message,
        intent=intent_result.intent.value,
        active_story=active_story,
        language=language,
        child_age=child_age,
        conversation_turn=1,  # First turn
        story_library=story_summaries  # All child's stories for context
    )

    # Emit conversation context event for the CompanionAgent's response
    await story_events.emit(
        "conversation_context", story_id_for_event,
        {
            "speaker": "narrator",
            "name": "Storyteller",
            "message": greeting.text[:300] + ("..." if len(greeting.text) > 300 else ""),
            "suggested_action": suggested_action
        }
    )

    return ConversationStartResponse(
        dialogue=greeting.text,
        audio=greeting.audio_base64,
        intent=intent_result.intent.value,
        suggested_action=suggested_action,
        active_story_id=child.active_story_id,
        child_name=child.name,
        child_age=child_age
    )


@router.post("/conversation/continue", response_model=ConversationStartResponse)
async def continue_conversation(request: ConversationStartRequest, conversation_turn: int = 2):
    """
    Continue an exploring conversation (for young children who need multiple turns).

    Same as /conversation/start but with conversation_turn parameter for
    tracking how many turns have passed. After 2-3 turns, CompanionAgent
    will gently suggest a story topic.
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    companion = get_companion_agent()
    if not companion:
        raise HTTPException(status_code=500, detail="CompanionAgent not initialized")

    # Get child profile
    child = await _coordinator.storage.get_child_profile(request.child_id)
    if not child:
        raise HTTPException(status_code=404, detail=f"Child profile not found: {request.child_id}")

    # Calculate child's age
    current_year = datetime.now().year
    child_age = current_year - child.birth_year

    # Get language
    language = request.language
    if not language or language == "en":
        parent = await _coordinator.storage.get_parent_account(child.parent_id)
        if parent and parent.language:
            language = parent.language

    # Get ALL child's stories
    all_stories = await _coordinator.storage.get_stories_by_child(request.child_id)
    story_summaries = []
    active_story = None
    for story in all_stories:
        if story.id == child.active_story_id:
            active_story = story
        summary = {
            "story_id": story.id,
            "title": story.structure.title if story.structure else "Untitled",
            "theme": story.structure.theme if story.structure else None,
            "is_active": story.id == child.active_story_id,
        }
        story_summaries.append(summary)

    # Check if child selected a specific story from their library by name
    selected_story_id = None
    message_lower = request.message.lower().strip()
    logger.info(f"🔍 Checking story name match in message: '{message_lower}'")
    logger.info(f"🔍 Story summaries: {[s.get('title', 'Untitled') for s in story_summaries]}")

    import re
    import unicodedata

    # PRIORITY 1: Check for NEW STORY intent FIRST (before continuation)
    # User wants to create a NEW story, not continue existing one
    # NOTE: Norwegian has grammatical gender - "ny" (common) vs "nytt" (neuter)
    # "en ny historie" but "et nytt eventyr"
    new_story_patterns = [
        r'(ny|nytt|new)\s*(fortelling|story|historie|eventyr)',  # "ny fortelling", "nytt eventyr", "new story"
        r'(annen|another|different)\s*(fortelling|story|historie|eventyr)',  # "annen historie"
        r'(lage|create|make)\s*(ny|nytt|new|en|et)',  # "lage ny", "lage et nytt", "create new"
        r'(ny|nytt|new)\s*(en|et|one)',  # "ny en", "nytt et", "new one"
        r'^(ny|nytt)[\s,!\.]*$',  # just "ny" or "nytt" by itself
        r'helt\s*(ny|nytt)',  # "helt nytt eventyr" (completely new)
        r'(ønsker|vil ha|want)\s*.{0,20}(ny|nytt|new)',  # "ønsker et nytt", "vil ha ny", "want a new"
    ]
    is_new_story = any(re.search(pattern, message_lower) for pattern in new_story_patterns)

    if is_new_story:
        logger.info(f"🆕 NEW STORY intent detected in: '{message_lower}' - NOT auto-selecting existing story")

    # PRIORITY 2: Check for affirmative OR continuation responses
    # These indicate user wants to continue their story
    affirmative_patterns = [
        r'^(ja|jo|japp|jepp|yes|yeah|yep|ok|okei)[\s,!\.]*',  # starts with yes
        r'^(den|det|that|that one|this one)[\s,!\.]*',  # "that one", "den"
        r'(hadde vært|would be|sounds good|høres bra)',  # confirmation phrases
    ]

    # Continuation intent patterns - user wants to continue the story
    continuation_patterns = [
        r'(fortsett|continue|videre|go on)',  # explicit continuation words
        r'(vil fortsette|want to continue)',  # phrases
        r'(historien|story|eventyret)',  # references "the story"
        r'(mer|more)',  # wants more
    ]

    is_affirmative = any(re.search(pattern, message_lower) for pattern in affirmative_patterns)
    is_continuation = any(re.search(pattern, message_lower) for pattern in continuation_patterns)

    # Auto-select active story ONLY if NOT asking for new story
    if is_new_story:
        # User wants new story - don't auto-select
        logger.info(f"🆕 Skipping auto-select due to NEW STORY intent")
    elif (is_affirmative or is_continuation) and active_story:
        selected_story_id = active_story.id
        logger.info(f"✅ Continuation intent detected, auto-selecting active story: {active_story.structure.title if active_story.structure else 'Unknown'}")
    elif is_affirmative and len(story_summaries) >= 1 and conversation_turn >= 2:
        # Fallback: if no active story but affirmative, select first story
        selected_story_id = story_summaries[0]["story_id"]
        logger.info(f"✅ Affirmative response detected, selecting first story: {story_summaries[0].get('title', 'Unknown')}")

    # Normalize unicode characters for comparison (handle Norwegian å, ø, æ)
    def normalize_text(text):
        # Keep the original text but normalize unicode representation
        return unicodedata.normalize('NFC', text.lower())

    message_normalized = normalize_text(message_lower)

    # SKIP title matching if user explicitly asked for NEW story
    if is_new_story:
        logger.info(f"🆕 Skipping story title matching due to NEW STORY intent")
    else:
        for summary in story_summaries:
            title = summary.get("title", "")
            if not title or len(title) < 3:
                continue

            title_lower = normalize_text(title)

            # Extract words, removing punctuation (like colons, commas)
            # Use regex to split on non-word characters, keeping Norwegian letters
            title_words = re.findall(r'[\wæøåÆØÅ]+', title_lower)
            # Keep words with 4+ characters for matching
            title_words_long = [w for w in title_words if len(w) >= 4]

            logger.info(f"🔍 Checking title '{title}' with words: {title_words_long}")

            # Check if any title word appears in message OR message word appears in title
            message_words = re.findall(r'[\wæøåÆØÅ]+', message_normalized)
            logger.info(f"🔍 Message words: {message_words}")

            # Match if any significant word matches
            matched = False
            for msg_word in message_words:
                if len(msg_word) >= 4:  # Only check words with 4+ chars
                    for title_word in title_words_long:
                        if msg_word in title_word or title_word in msg_word:
                            logger.info(f"🔍 Match found: '{msg_word}' <-> '{title_word}'")
                            matched = True
                            break
                if matched:
                    break

            if matched or title_lower in message_normalized:
                selected_story_id = summary["story_id"]
                logger.info(f"📖 Child selected story by name: {summary['title']} ({selected_story_id})")
                break

    # If story was selected, return resume action with that story
    if selected_story_id:
        # Find the full story object for greeting
        selected_story = next((s for s in all_stories if s.id == selected_story_id), None)

        # Check chapter 1 status and trigger generation/audio EARLY (before greeting)
        should_trigger_chapter = False
        should_trigger_audio = False
        if selected_story:
            story_has_structure = (
                selected_story.structure and
                selected_story.structure.chapters and
                len(selected_story.structure.chapters) > 0
            )
            chapter_1_exists = any(
                ch.content and len(ch.content) > 100
                for ch in (selected_story.chapters or [])
                if ch.number == 1
            )

            if story_has_structure and not chapter_1_exists:
                should_trigger_chapter = True
                logger.info(f"📝 Story selected - triggering chapter 1 generation EARLY for {selected_story_id}")
            elif chapter_1_exists:
                should_trigger_audio = True
                logger.info(f"🔊 Story selected - triggering chapter 1 audio EARLY for {selected_story_id}")

        # EARLY TRIGGER: Start chapter/audio generation NOW (runs in background while greeting generates)
        if should_trigger_chapter:
            logger.info(f"🚀 [EARLY] Starting chapter 1 generation for {selected_story_id}")
            asyncio.create_task(_trigger_chapter_generation_from_routes(selected_story_id, 1))
        elif should_trigger_audio:
            logger.info(f"🔊 [EARLY] Starting chapter 1 audio generation for {selected_story_id}")
            asyncio.create_task(_trigger_audio_generation_from_routes(selected_story_id, 1))

        # Generate a confirmation greeting (audio generation is already running in background!)
        # DON'T pass full story_library here - user already made their choice!
        # Only pass the selected story so CompanionAgent doesn't ask "which story?" again
        selected_story_summary = next((s for s in story_summaries if s["story_id"] == selected_story_id), None)
        greeting = await companion.generate_greeting(
            child_name=child.name,
            message=request.message,
            intent="continue",
            active_story=selected_story,
            language=language,
            child_age=child_age,
            conversation_turn=conversation_turn,
            story_library=[selected_story_summary] if selected_story_summary else []  # Only the selected story!
        )

        # Emit conversation context events for Observatory
        story_id_for_event = selected_story_id or "pre_story"
        await story_events.emit(
            "conversation_context", story_id_for_event,
            {
                "speaker": "user",
                "name": child.name,
                "message": request.message,
                "intent": "continue",
                "turn": conversation_turn
            }
        )
        await story_events.emit(
            "conversation_context", story_id_for_event,
            {
                "speaker": "narrator",
                "name": "Storyteller",
                "message": greeting.text[:300] + ("..." if len(greeting.text) > 300 else ""),
                "suggested_action": "resume_story"
            }
        )

        return ConversationStartResponse(
            dialogue=greeting.text,
            audio=greeting.audio_base64,
            intent="continue",
            suggested_action="resume_story",
            active_story_id=selected_story_id,  # The SELECTED story, not the profile's active
            child_name=child.name,
            child_age=child_age
        )

    # Re-classify intent (child may have changed their mind)
    intent_result = await classify_conversation_intent(
        message=request.message,
        has_active_story=active_story is not None,
        child_age=child_age,
        language=language
    )

    # Override intent if we detected NEW STORY earlier (before classifier ran)
    final_intent = intent_result.intent.value
    if is_new_story:
        final_intent = "new_story"
        logger.info(f"🆕 Overriding intent to 'new_story' (detected 'ny fortelling' etc. in message)")

    logger.info(f"🎯 Continue conversation - Intent: {final_intent}, Turn: {conversation_turn}")

    # Limit story_library to prevent re-asking "which story?"
    # If there's an active story, only show that one (or none if exploring)
    limited_story_library = []
    if active_story:
        active_summary = next((s for s in story_summaries if s["story_id"] == active_story.id), None)
        if active_summary:
            limited_story_library = [active_summary]

    # Generate response with turn tracking
    greeting = await companion.generate_greeting(
        child_name=child.name,
        message=request.message,
        intent=final_intent,
        active_story=active_story,
        language=language,
        child_age=child_age,
        conversation_turn=conversation_turn,
        story_library=limited_story_library  # Only active story, not ALL stories
    )

    # Emit conversation context events for Observatory
    story_id_for_event = child.active_story_id or "pre_story"
    suggested_action = "create_story" if is_new_story else intent_result.suggested_action
    await story_events.emit(
        "conversation_context", story_id_for_event,
        {
            "speaker": "user",
            "name": child.name,
            "message": request.message,
            "intent": final_intent,
            "turn": conversation_turn
        }
    )
    await story_events.emit(
        "conversation_context", story_id_for_event,
        {
            "speaker": "narrator",
            "name": "Storyteller",
            "message": greeting.text[:300] + ("..." if len(greeting.text) > 300 else ""),
            "suggested_action": suggested_action
        }
    )

    return ConversationStartResponse(
        dialogue=greeting.text,
        audio=greeting.audio_base64,
        intent=final_intent,
        suggested_action=suggested_action,
        active_story_id=child.active_story_id,
        child_name=child.name,
        child_age=child_age
    )


# ===== Story Initialization =====

@router.post("/conversation/init", response_model=Dict[str, Any])
async def initialize_story(
    request: StoryInitRequest,
    background_tasks: BackgroundTasks
):
    """
    Initialize a new story.

    This endpoint:
    1. Creates story in Firebase
    2. Returns immediate dialogue response
    3. Kicks off background story generation

    Request body:
    {
        "prompt": "Tell me a story about Vikings",
        "type": "historical",
        "language": "en",
        "target_age": 8,
        "preferences": {
            "educational_focus": "history",
            "difficulty": "medium",
            "themes": ["exploration", "courage"],
            "scary_level": "mild"
        }
    }

    Response:
    {
        "success": true,
        "story_id": "story_abc123",
        "welcome_message": "Oh wow! Vikings! What an exciting choice!...",
        "story": { ... }
    }
    """
    # Debug: Log request details
    logger.info(f"📝 Story init: child_id={request.child_id}, age={request.target_age}, lang={request.language}, chapters={request.chapters_to_write}")

    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        # Initialize story (gets immediate dialogue response)
        # New family account model: child_id takes precedence over user_id
        result = await _coordinator.initialize_story(
            prompt=request.prompt,
            user_id=request.user_id,  # DEPRECATED: Kept for backward compat
            target_age=request.target_age,
            preferences=request.preferences.model_dump() if request.preferences else None,
            child_id=request.child_id,  # New: Uses child profile for age/language
            language=request.language,   # New: Can override parent's language
        )

        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to initialize story"))

        story_id = result["story_id"]

        # Queue pre-story conversation messages for Chapter 1
        if request.pre_story_messages and len(request.pre_story_messages) > 0:
            logger.info(f"📝 Queueing {len(request.pre_story_messages)} pre-story messages for Chapter 1")
            for msg in request.pre_story_messages:
                if msg and msg.strip():  # Skip empty messages
                    queued_input = QueuedInput(
                        id=str(uuid.uuid4()),
                        tier=InputTier.TIER_2_PREFERENCE,
                        raw_input=msg.strip(),
                        classified_intent="pre_story_preference",
                        story_direction=msg.strip(),  # Use message as story direction
                        target_chapter=1,
                        created_at=datetime.now()
                    )
                    await _coordinator.storage.queue_user_input(story_id, queued_input)
                    logger.info(f"   ✅ Queued: {msg[:50]}...")

            # Emit event for Observatory
            await story_events.emit("pre_story_inputs_queued", story_id, {
                "count": len(request.pre_story_messages),
                "target_chapter": 1
            })

        # Background task: Generate structure and characters
        background_tasks.add_task(
            _generate_story_background,
            story_id,
            request.prompt,
            request.chapters_to_write
        )

        return result

    except ValidationError as e:
        # Handle Pydantic validation errors with user-friendly messages
        error_details = e.errors()
        logger.warning(f"⚠️ Pydantic validation errors: {error_details}")

        for err in error_details:
            if err.get("type") == "string_too_long":
                # Get full location tuple and check if it involves "prompt"
                loc = err.get("loc", ["unknown"])
                field = loc[0] if loc else "unknown"
                is_prompt = "prompt" in str(loc)

                # Debug logging
                logger.warning(f"⚠️ String too long error: loc={loc}, field={field}, is_prompt={is_prompt}")

                # Determine actual length - handle both prompt and pre_story_messages
                if is_prompt or field == "prompt":
                    actual_length = len(request.prompt) if hasattr(request, 'prompt') else 0
                    max_length = PROMPT_MAX_LENGTH
                    field_name = "story prompt"
                elif field == "pre_story_messages" or "pre_story_messages" in str(loc):
                    # Individual message in pre_story_messages list
                    idx = loc[1] if len(loc) > 1 and isinstance(loc[1], int) else 0
                    actual_length = len(request.pre_story_messages[idx]) if request.pre_story_messages and len(request.pre_story_messages) > idx else 0
                    max_length = PROMPT_MAX_LENGTH  # Allow same length as prompt for conversation messages
                    field_name = "conversation message"
                elif field == "raw_input":
                    # QueuedInput.raw_input - queued pre-story messages
                    actual_length = 0  # We don't have direct access to the failing input here
                    max_length = PROMPT_MAX_LENGTH  # Same limit as prompt
                    field_name = "conversation message"
                else:
                    actual_length = 0
                    max_length = MESSAGE_MAX_LENGTH
                    field_name = str(field)

                logger.warning(f"⚠️ Validation error: {field_name} too long ({actual_length} chars, max {max_length})")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "content_too_long",
                        "field": str(field),
                        "message": f"Your {field_name} is too long ({actual_length:,} characters). The maximum is {max_length:,} characters. Try shortening your reference material or splitting it into key points.",
                        "actual_length": actual_length,
                        "max_length": max_length
                    }
                )
        # Generic validation error
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")

    except Exception as e:
        import traceback
        print(f"❌ ERROR in initialize_story:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error initializing story: {str(e)}")


async def _generate_story_background(story_id: str, prompt: str, chapters_to_write: int = None):
    """
    Background task to generate story structure and first chapter.

    Handles error logging, Firebase status updates, and event emissions for real-time feedback.
    """
    try:
        # AUDIO FIX: Wait 1 second for WebSocket to connect before emitting events
        await asyncio.sleep(1)
        logger.info(f"🚀 Starting background story generation for {story_id}")

        # Activate CompanionAgent for proactive engagement during wait
        companion = get_companion_agent()
        if companion:
            await companion.activate(story_id, {"prompt": prompt})
            logger.info(f"🤝 CompanionAgent activated for {story_id}")

        # ============================================================================
        # STEP 1: Generate Structure
        # ============================================================================
        try:
            structure_result = await _coordinator.generate_story_structure(story_id, dialogue_context=prompt)
            if not structure_result["success"]:
                error_msg = structure_result.get('error', 'Unknown error')
                logger.error(f"❌ Failed to generate structure for {story_id}: {error_msg}")

                # Update story status to FAILED in Firebase
                await _coordinator.storage.update_story_status(story_id, StoryStatus.FAILED.value)

                # Emit error event for real-time feedback
                await story_events.emit("error", story_id, {
                    "stage": "structure_generation",
                    "error": error_msg,
                    "message": "Failed to create story structure. Please try again."
                })
                return

        except Exception as e:
            logger.error("BackgroundStory", f"Exception during structure generation for {story_id}", e)
            await _coordinator.storage.update_story_status(story_id, StoryStatus.FAILED.value)
            await story_events.emit("error", story_id, {
                "stage": "structure_generation",
                "error": str(e),
                "message": "An unexpected error occurred while creating the story structure."
            })
            return

        # ============================================================================
        # STEP 2: Create Characters
        # ============================================================================
        try:
            characters_result = await _coordinator.create_characters(story_id)
            if not characters_result["success"]:
                error_msg = characters_result.get('error', 'Unknown error')
                logger.error(f"❌ Failed to create characters for {story_id}: {error_msg}")

                # Update story status to FAILED
                await _coordinator.storage.update_story_status(story_id, StoryStatus.FAILED.value)

                # Emit error event
                await story_events.emit("error", story_id, {
                    "stage": "character_creation",
                    "error": error_msg,
                    "message": "Failed to create story characters. The story structure is ready, but character creation failed."
                })
                return

        except Exception as e:
            logger.error("BackgroundStory", f"Exception during character creation for {story_id}", e)
            await _coordinator.storage.update_story_status(story_id, StoryStatus.FAILED.value)
            await story_events.emit("error", story_id, {
                "stage": "character_creation",
                "error": str(e),
                "message": "An unexpected error occurred while creating characters."
            })
            return

        # ============================================================================
        # STEP 3: Write Chapter 1 (One-Chapter-At-A-Time)
        # ============================================================================
        # NEW: Only write Chapter 1 during initialization
        # Future chapters are generated on-demand when the user starts playing
        # the current chapter (prefetch behavior)
        settings = get_settings()
        num_chapters_planned = len(structure_result["structure"].get("chapters", []))

        # Use initial_chapters_to_write (default 1) instead of beta_chapters_to_write
        # This can be overridden by request for backwards compatibility
        chapters_limit = chapters_to_write if chapters_to_write is not None else settings.initial_chapters_to_write
        chapters_to_write_count = min(chapters_limit, num_chapters_planned)

        logger.info(f"📚 Writing {chapters_to_write_count} of {num_chapters_planned} planned chapters (one-chapter-at-a-time mode)...")

        chapters_completed = 0
        for chapter_num in range(1, chapters_to_write_count + 1):
            try:
                logger.info(f"   ✍️  Writing chapter {chapter_num}/{chapters_to_write_count}...")
                logger.info(f"   📊 Pre-chapter status check:")

                # PRE-FLIGHT CHECKS: Verify data availability
                try:
                    story_check = await _coordinator.storage.get_story(story_id)
                    if not story_check:
                        raise ValueError(f"Story {story_id} not found in Firebase")
                    logger.info(f"      ✅ Story data available")

                    if not story_check.structure:
                        raise ValueError(f"Story structure missing for {story_id}")
                    logger.info(f"      ✅ Story structure available ({len(story_check.structure.chapters)} chapters)")

                    characters_check = await _coordinator.storage.get_characters(story_id)
                    logger.info(f"      ✅ Characters available: {len(characters_check)} characters")

                    if len(characters_check) == 0 and len(story_check.structure.characters_needed) > 0:
                        raise ValueError(f"No characters found but {len(story_check.structure.characters_needed)} expected")

                except Exception as pre_check_error:
                    logger.error(f"   ❌ Pre-flight check failed: {pre_check_error}")
                    await story_events.emit("error", story_id, {
                        "stage": "chapter_writing_preflight",
                        "chapter_number": chapter_num,
                        "error": str(pre_check_error),
                        "message": f"Pre-flight check failed before writing chapter {chapter_num}: {str(pre_check_error)}"
                    })
                    raise

                # Execute chapter writing
                logger.info(f"   🚀 Calling coordinator.write_chapter({story_id}, {chapter_num})...")
                chapter_result = await _coordinator.write_chapter(story_id, chapter_num)
                logger.info(f"   📥 Received result from write_chapter: {chapter_result.get('success', 'UNKNOWN')}")

                if not chapter_result["success"]:
                    error_msg = chapter_result.get('error', 'Unknown error')
                    logger.error(f"   ❌ Failed to write chapter {chapter_num}: {error_msg}")
                    logger.error(f"   📊 Full error result: {chapter_result}")

                    # Don't mark story as FAILED - partial completion is acceptable
                    # Emit error event for this specific chapter
                    await story_events.emit("error", story_id, {
                        "stage": "chapter_writing",
                        "chapter_number": chapter_num,
                        "error": error_msg,
                        "message": f"Failed to write chapter {chapter_num}. {chapters_completed} chapters completed successfully."
                    })
                    break

                chapters_completed += 1
                logger.info(f"   ✅ Chapter {chapter_num} completed! Total completed: {chapters_completed}")

            except Exception as e:
                logger.error("BackgroundStory", f"EXCEPTION during chapter {chapter_num} writing", e)
                logger.error(f"   📊 Exception type: {type(e).__name__}")
                logger.error(f"   📊 Exception details: {str(e)}")

                # Try to get more context about the failure
                try:
                    import traceback
                    tb_str = traceback.format_exc()
                    logger.error(f"   📊 Full traceback:\n{tb_str}")
                except:
                    pass

                await story_events.emit("error", story_id, {
                    "stage": "chapter_writing",
                    "chapter_number": chapter_num,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "message": f"An unexpected error occurred while writing chapter {chapter_num}: {type(e).__name__}"
                })
                break

        # ============================================================================
        # COMPLETION (Chapter 1 Ready - not full story)
        # ============================================================================
        if chapters_completed > 0:
            logger.info(f"✅ Chapter 1 ready for story {story_id}! ({num_chapters_planned} chapters planned)")
            # Update story status to COMPLETED (Chapter 1 ready)
            # Note: Future chapters generated on-demand via prefetch
            await _coordinator.storage.update_story_status(story_id, StoryStatus.COMPLETED.value)
            await story_events.emit("complete", story_id, {
                "chapters_completed": chapters_completed,
                "chapters_planned": num_chapters_planned,
                "message": f"Chapter 1 is ready! {num_chapters_planned - 1} more chapters await your adventure.",
                "one_chapter_at_a_time": True  # Signal to frontend
            })
        else:
            logger.warning(f"⚠️  Story {story_id} structure and characters created, but no chapters were written.")

    except Exception as e:
        # Catch-all for any unexpected errors
        logger.error("BackgroundStory", f"CRITICAL: Background story generation failed for {story_id}", e)
        try:
            await _coordinator.storage.update_story_status(story_id, StoryStatus.FAILED.value)
            await story_events.emit("error", story_id, {
                "stage": "unknown",
                "error": str(e),
                "message": "An unexpected critical error occurred during story generation."
            })
        except Exception as nested_e:
            logger.error("BackgroundStory", f"Failed to report error for {story_id}", nested_e)


# NOTE: Conversation messages are handled via WebSocket (/ws/story/{story_id})
# See src/api/websocket.py for real-time dialogue handling


@router.get("/stories/{story_id}")
async def get_story(story_id: str):
    """
    Get story details.

    Returns complete story data including:
    - Metadata
    - Structure
    - Characters
    - Chapters
    - Dialogue history
    - Educational progress
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        story = await _coordinator.storage.get_story(story_id)

        if not story:
            raise HTTPException(status_code=404, detail="Story not found")

        return {
            "success": True,
            "story": story.model_dump()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching story: {str(e)}")


@router.get("/stories/{story_id}/chapters/{chapter_number}")
async def get_chapter(story_id: str, chapter_number: int):
    """Get a specific chapter"""
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        chapter = await _coordinator.storage.get_chapter(story_id, chapter_number)

        if not chapter:
            raise HTTPException(status_code=404, detail="Chapter not found")

        return {
            "success": True,
            "chapter": chapter.model_dump()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chapter: {str(e)}")


@router.post("/stories/{story_id}/chapters/{chapter_number}/generate")
async def generate_chapter(story_id: str, chapter_number: int, background_tasks: BackgroundTasks):
    """
    Generate a specific chapter.

    This can be called to generate subsequent chapters after the first.
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    # Add to background tasks
    background_tasks.add_task(
        _coordinator.write_chapter,
        story_id,
        chapter_number
    )

    return {
        "success": True,
        "message": f"Chapter {chapter_number} generation started",
        "story_id": story_id,
        "chapter_number": chapter_number
    }


@router.post("/stories/{story_id}/generate-all")
async def generate_all_chapters(story_id: str):
    """
    Generate ALL chapters for a story immediately.

    Unlike buffered generation (which generates chapters on-demand as user reads),
    this endpoint generates all chapters in sequence before returning.

    Use this for:
    - E2E testing (validate complete story generation)
    - When you want the full story available upfront
    - Debugging story generation issues

    Note: This is a long-running operation (can take 10+ minutes for a full story).
    Consider using WebSocket for progress updates.
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        result = await _coordinator.generate_all_chapters(story_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chapters: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Lore Lantern",
        "coordinator_initialized": _coordinator is not None
    }


@router.get("/debug/rate-limiter")
async def get_rate_limiter_stats():
    """
    Get adaptive rate limiter statistics.

    Returns per-provider concurrency info, success rates, and scaling history.
    Useful for monitoring and tuning the rate limiting strategy.
    """
    from src.services.adaptive_rate_limiter import get_rate_limiter
    limiter = get_rate_limiter()
    stats = limiter.get_stats()

    return {
        "status": "ok",
        "rate_limiter": {
            "providers": stats,
            "summary": {
                "total_providers": len(stats),
                "total_calls": sum(p.get("total_calls", 0) for p in stats.values()),
                "total_rate_limits": sum(p.get("rate_limited_calls", 0) for p in stats.values()),
            }
        }
    }


# ============================================================================
# User Profile Endpoints
# ============================================================================

@router.post("/users", response_model=Dict[str, Any])
async def create_user_profile(
    user_id: str,
    display_name: str,
    current_age: int
):
    """
    Create a new user profile.

    Request body:
    {
        "user_id": "emma_7",
        "display_name": "Emma",
        "current_age": 7
    }
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        # Check if user already exists
        existing_profile = await _coordinator.storage.get_user_profile(user_id)
        if existing_profile:
            raise HTTPException(status_code=400, detail=f"User profile '{user_id}' already exists")

        # Create new profile
        from src.models import UserProfile
        profile = UserProfile(
            user_id=user_id,
            display_name=display_name,
            current_age=current_age
        )

        # Save to Firebase
        saved_profile = await _coordinator.storage.save_user_profile(profile)

        return {
            "success": True,
            "message": f"User profile created for {display_name}",
            "profile": saved_profile.model_dump()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating user profile: {str(e)}")


@router.get("/users/{user_id}", response_model=Dict[str, Any])
async def get_user_profile(user_id: str):
    """
    Get user profile with stats and progression.

    Returns:
    {
        "success": true,
        "profile": {
            "user_id": "emma_7",
            "display_name": "Emma",
            "current_age": 7,
            "total_stories_completed": 5,
            "reading_level": 2,
            "curiosity_score": 12,
            ...
        }
    }
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        profile = await _coordinator.storage.get_user_profile(user_id)

        if not profile:
            raise HTTPException(status_code=404, detail=f"User profile '{user_id}' not found")

        return {
            "success": True,
            "profile": profile.model_dump()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user profile: {str(e)}")


@router.get("/users/{user_id}/stories", response_model=Dict[str, Any])
async def get_user_stories(user_id: str):
    """
    Get all stories for a user (their story library).

    Returns:
    {
        "success": true,
        "user_id": "emma_7",
        "stories": [
            {
                "id": "story_abc123",
                "prompt": "Tell me about Vikings",
                "status": "completed",
                "chapters": 5,
                ...
            }
        ]
    }
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        # Get user's stories from Firebase
        stories = await _coordinator.storage.get_stories_by_user(user_id)

        return {
            "success": True,
            "user_id": user_id,
            "total_stories": len(stories),
            "stories": [story.model_dump() for story in stories]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user stories: {str(e)}")


# ============================================================================
# Parent Account Endpoints (New Family Account System)
# ============================================================================

@router.post("/parents", response_model=Dict[str, Any])
async def create_parent_account(request: ParentAccountCreate):
    """
    Create a new parent account (Lantern owner).

    Request body:
    {
        "language": "en",
        "display_name": "Smith Family"
    }

    Response:
    {
        "success": true,
        "parent_id": "parent_abc123",
        "message": "Parent account created",
        "account": { ...full parent account... }
    }
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        # Create new parent account (use provided parent_id if given)
        parent_kwargs = {
            "language": request.language,
            "display_name": request.display_name
        }
        if request.parent_id:
            parent_kwargs["parent_id"] = request.parent_id

        parent = ParentAccount(**parent_kwargs)

        # Save to Firebase
        saved_parent = await _coordinator.storage.save_parent_account(parent)

        return {
            "success": True,
            "parent_id": saved_parent.parent_id,
            "message": f"Parent account created: {saved_parent.display_name or saved_parent.parent_id}",
            "account": saved_parent.model_dump()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating parent account: {str(e)}")


@router.get("/parents", response_model=Dict[str, Any])
async def list_parent_accounts():
    """
    List all parent accounts.

    Response:
    {
        "success": true,
        "parents": [{ ...parent account... }, ...]
    }
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        parents = await _coordinator.storage.list_parent_accounts()
        return {
            "success": True,
            "parents": [p.model_dump() for p in parents]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing parent accounts: {str(e)}")


@router.get("/parents/{parent_id}", response_model=Dict[str, Any])
async def get_parent_account(parent_id: str):
    """
    Get parent account details.

    Response:
    {
        "success": true,
        "account": {
            "parent_id": "parent_abc123",
            "language": "en",
            "child_ids": ["child_xyz789", "child_abc456"],
            "display_name": "Smith Family"
        }
    }
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        parent = await _coordinator.storage.get_parent_account(parent_id)

        if not parent:
            raise HTTPException(status_code=404, detail=f"Parent account '{parent_id}' not found")

        return {
            "success": True,
            "account": parent.model_dump()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching parent account: {str(e)}")


@router.patch("/parents/{parent_id}", response_model=Dict[str, Any])
async def update_parent_account(parent_id: str, request: ParentAccountUpdate):
    """
    Update parent account (e.g., change language).

    Request body:
    {
        "language": "no"
    }

    Response:
    {
        "success": true,
        "message": "Parent account updated",
        "account": { ...updated account... }
    }
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        updates = {k: v for k, v in request.model_dump().items() if v is not None}
        updated_parent = await _coordinator.storage.update_parent_account(parent_id, updates)

        if not updated_parent:
            raise HTTPException(status_code=404, detail=f"Parent account '{parent_id}' not found")

        return {
            "success": True,
            "message": "Parent account updated",
            "account": updated_parent.model_dump()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating parent account: {str(e)}")


# ============================================================================
# Child Profile Endpoints
# ============================================================================

@router.post("/parents/{parent_id}/children", response_model=Dict[str, Any])
async def add_child_profile(parent_id: str, request: ChildProfileCreate):
    """
    Add a new child profile to a parent account.

    Request body:
    {
        "name": "Emma",
        "birth_year": 2018
    }

    Response:
    {
        "success": true,
        "child_id": "child_xyz789",
        "message": "Child profile created for Emma",
        "profile": {
            "child_id": "child_xyz789",
            "name": "Emma",
            "current_age": 7,
            "story_count": 0
        }
    }
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        # Verify parent exists
        parent = await _coordinator.storage.get_parent_account(parent_id)
        if not parent:
            raise HTTPException(status_code=404, detail=f"Parent account '{parent_id}' not found")

        # Create child profile (use provided child_id if given)
        child_kwargs = {
            "parent_id": parent_id,
            "name": request.name,
            "birth_year": request.birth_year
        }
        if request.child_id:
            child_kwargs["child_id"] = request.child_id

        child = ChildProfile(**child_kwargs)

        # Save child profile
        saved_child = await _coordinator.storage.save_child_profile(child)

        # Add child to parent's child_ids
        parent.child_ids.append(saved_child.child_id)
        await _coordinator.storage.save_parent_account(parent)

        # Create empty learning progress for child
        progress = LearningProgress(child_id=saved_child.child_id)
        await _coordinator.storage.save_learning_progress(progress)

        return {
            "success": True,
            "child_id": saved_child.child_id,
            "message": f"Child profile created for {saved_child.name}",
            "profile": {
                "child_id": saved_child.child_id,
                "name": saved_child.name,
                "current_age": saved_child.current_age,
                "story_count": len(saved_child.story_ids),
                "active_story_id": saved_child.active_story_id
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating child profile: {str(e)}")


@router.get("/parents/{parent_id}/children", response_model=Dict[str, Any])
async def list_children(parent_id: str):
    """
    List all children for a parent account.

    Response:
    {
        "success": true,
        "parent_id": "parent_abc123",
        "children": [
            {
                "child_id": "child_xyz789",
                "name": "Emma",
                "current_age": 7,
                "story_count": 5,
                "active_story_id": "story_abc123"
            }
        ]
    }
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        # Verify parent exists
        parent = await _coordinator.storage.get_parent_account(parent_id)
        if not parent:
            raise HTTPException(status_code=404, detail=f"Parent account '{parent_id}' not found")

        # Get children
        children = await _coordinator.storage.get_children_by_parent(parent_id)

        return {
            "success": True,
            "parent_id": parent_id,
            "children": [
                {
                    "child_id": child.child_id,
                    "name": child.name,
                    "current_age": child.current_age,
                    "story_count": len(child.story_ids),
                    "active_story_id": child.active_story_id
                }
                for child in children
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing children: {str(e)}")


@router.get("/children/{child_id}", response_model=Dict[str, Any])
async def get_child_profile(child_id: str):
    """
    Get a specific child profile.

    Response:
    {
        "success": true,
        "profile": {
            "child_id": "child_xyz789",
            "name": "Emma",
            "current_age": 7,
            "story_count": 5,
            "active_story_id": "story_abc123"
        }
    }

    NOTE: Does NOT include learning_progress data (backend-only).
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        child = await _coordinator.storage.get_child_profile(child_id)

        if not child:
            raise HTTPException(status_code=404, detail=f"Child profile '{child_id}' not found")

        return {
            "success": True,
            "profile": {
                "child_id": child.child_id,
                "name": child.name,
                "current_age": child.current_age,
                "story_count": len(child.story_ids),
                "active_story_id": child.active_story_id,
                "created_at": child.created_at.isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching child profile: {str(e)}")


@router.patch("/children/{child_id}", response_model=Dict[str, Any])
async def update_child_profile(child_id: str, request: ChildProfileUpdate):
    """
    Update a child profile (name or birth_year).

    Request body:
    {
        "name": "Emma Rose"
    }

    Response:
    {
        "success": true,
        "message": "Child profile updated",
        "profile": { ...updated profile... }
    }
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        updates = {k: v for k, v in request.model_dump().items() if v is not None}
        updated_child = await _coordinator.storage.update_child_profile(child_id, updates)

        if not updated_child:
            raise HTTPException(status_code=404, detail=f"Child profile '{child_id}' not found")

        return {
            "success": True,
            "message": "Child profile updated",
            "profile": {
                "child_id": updated_child.child_id,
                "name": updated_child.name,
                "current_age": updated_child.current_age,
                "story_count": len(updated_child.story_ids),
                "active_story_id": updated_child.active_story_id
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating child profile: {str(e)}")


@router.delete("/children/{child_id}", response_model=Dict[str, Any])
async def remove_child_profile(child_id: str, delete_learning: bool = True):
    """
    Remove a child profile.

    NOTE: This is a soft delete. Stories are preserved but unlinked.
    Learning progress is deleted for GDPR compliance unless delete_learning=False.

    Response:
    {
        "success": true,
        "message": "Child profile removed",
        "learning_data_deleted": true
    }
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        child = await _coordinator.storage.get_child_profile(child_id)
        if not child:
            raise HTTPException(status_code=404, detail=f"Child profile '{child_id}' not found")

        stories_count = len(child.story_ids)
        deleted = await _coordinator.storage.delete_child_profile(child_id, delete_learning=delete_learning)

        if not deleted:
            raise HTTPException(status_code=500, detail="Failed to delete child profile")

        return {
            "success": True,
            "message": f"Child profile '{child.name}' removed",
            "stories_preserved": stories_count,
            "learning_data_deleted": delete_learning
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing child profile: {str(e)}")


# ============================================================================
# Story Library Endpoints (Child-Centric)
# ============================================================================

@router.get("/children/{child_id}/stories", response_model=Dict[str, Any])
async def get_child_story_library(child_id: str):
    """
    Get all stories for a child (their story library).

    Response:
    {
        "success": true,
        "child_id": "child_xyz789",
        "total_stories": 5,
        "stories": [
            {
                "story_id": "story_abc123",
                "title": "Leif's Iceland Adventure",
                "prompt": "Tell me about Vikings...",
                "status": "paused",
                "chapters_completed": 2,
                "total_chapters": 5,
                "can_continue": true
            }
        ]
    }
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        child = await _coordinator.storage.get_child_profile(child_id)
        if not child:
            raise HTTPException(status_code=404, detail=f"Child profile '{child_id}' not found")

        stories = await _coordinator.storage.get_stories_by_child(child_id)

        story_items = []
        for story in stories:
            chapters_completed = len([ch for ch in story.chapters if ch.content]) if story.chapters else 0
            total_chapters = len(story.structure.chapters) if story.structure else 0
            can_continue = story.status.value not in ["completed", "failed"]

            story_items.append({
                "story_id": story.id,
                "title": story.structure.title if story.structure else story.prompt[:50],
                "prompt": story.prompt,
                "status": story.status.value,
                "chapters_completed": chapters_completed,
                "total_chapters": total_chapters,
                "can_continue": can_continue,
                "created_at": story.metadata.created_at.isoformat() if story.metadata.created_at else None
            })

        return {
            "success": True,
            "child_id": child_id,
            "total_stories": len(story_items),
            "stories": story_items
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching story library: {str(e)}")


@router.post("/children/{child_id}/stories/{story_id}/continue", response_model=Dict[str, Any])
async def continue_story(child_id: str, story_id: str):
    """
    Continue a paused story.

    - Loads the story state
    - Sets as active_story_id on child profile
    - Returns current chapter position

    Response:
    {
        "success": true,
        "story_id": "story_abc123",
        "title": "Leif's Iceland Adventure",
        "current_chapter": 2,
        "chapter_status": "ready",
        "message": "Resuming from Chapter 2: Storm at Sea"
    }
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        # Verify child exists
        child = await _coordinator.storage.get_child_profile(child_id)
        if not child:
            raise HTTPException(status_code=404, detail=f"Child profile '{child_id}' not found")

        # Verify story exists and belongs to child
        story = await _coordinator.storage.get_story(story_id)
        if not story:
            raise HTTPException(status_code=404, detail=f"Story '{story_id}' not found")

        if story.metadata.child_id != child_id and story.metadata.user_id != child_id:
            raise HTTPException(status_code=403, detail="Story does not belong to this child")

        # Get reading state
        reading_state = await _coordinator.storage.get_reading_state(story_id)
        # Ensure current_chapter is at least 1 (chapters are 1-indexed)
        current_chapter = reading_state.current_chapter if reading_state and reading_state.current_chapter else 1
        current_chapter = max(1, current_chapter)  # Safeguard against chapter 0
        chapter_status = "ready"

        if reading_state and reading_state.chapter_statuses:
            ch_status = reading_state.chapter_statuses.get(str(current_chapter))
            if ch_status:
                chapter_status = ch_status.value if hasattr(ch_status, 'value') else str(ch_status)

        # Set as active story
        await _coordinator.storage.set_active_story(child_id, story_id)

        # Get chapter title
        chapter_title = f"Chapter {current_chapter}"
        if story.structure and story.structure.chapters:
            for ch in story.structure.chapters:
                if ch.number == current_chapter:
                    chapter_title = ch.title
                    break

        # AUTO-TRIGGER chapter generation OR audio generation if needed
        story_has_structure = (
            story.structure and
            story.structure.chapters and
            len(story.structure.chapters) > 0
        )

        # Find current chapter
        current_chapter_obj = None
        for ch in (story.chapters or []):
            if ch.number == current_chapter:
                current_chapter_obj = ch
                break

        chapter_has_content = current_chapter_obj and current_chapter_obj.content and len(current_chapter_obj.content) > 100

        if story_has_structure and not chapter_has_content:
            # Chapter needs to be written
            # NOTE: EARLY trigger in /conversation/continue should have already started this
            # Only trigger if not already running (fallback for direct API calls)
            logger.info(f"🚀 [continue_story] Chapter {current_chapter} needs generation for {story_id} (may already be running)")
            asyncio.create_task(_trigger_chapter_generation_from_routes(story_id, current_chapter))
            chapter_status = "generating"
        elif chapter_has_content:
            # Chapter exists - audio generation should already be triggered by EARLY trigger
            # Skip duplicate trigger to save API credits
            logger.info(f"🔊 [continue_story] Chapter {current_chapter} exists for {story_id} - audio should already be generating (EARLY trigger)")
            chapter_status = "generating_audio"

        return {
            "success": True,
            "story_id": story_id,
            "title": story.structure.title if story.structure else story.prompt[:50],
            "current_chapter": current_chapter,
            "chapter_status": chapter_status,
            "message": f"Resuming from Chapter {current_chapter}: {chapter_title}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error continuing story: {str(e)}")


@router.get("/children/{child_id}/active-story", response_model=Dict[str, Any])
async def get_active_story(child_id: str):
    """
    Get the currently active story for a child (quick resume).

    Response (with active story):
    {
        "success": true,
        "has_active_story": true,
        "story": {
            "story_id": "story_abc123",
            "title": "Leif's Iceland Adventure",
            "current_chapter": 2,
            "chapter_ready": true
        }
    }

    Response (no active story):
    {
        "success": true,
        "has_active_story": false,
        "story": null
    }
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        child = await _coordinator.storage.get_child_profile(child_id)
        if not child:
            raise HTTPException(status_code=404, detail=f"Child profile '{child_id}' not found")

        if not child.active_story_id:
            return {
                "success": True,
                "has_active_story": False,
                "story": None
            }

        # Get story details
        story = await _coordinator.storage.get_story(child.active_story_id)
        if not story:
            # Active story was deleted, clear reference
            await _coordinator.storage.set_active_story(child_id, None)
            return {
                "success": True,
                "has_active_story": False,
                "story": None
            }

        # Get reading state
        reading_state = await _coordinator.storage.get_reading_state(child.active_story_id)
        current_chapter = reading_state.current_chapter if reading_state else 1

        # Check if chapter is ready
        chapter_ready = False
        if story.chapters:
            for ch in story.chapters:
                if ch.chapter_number == current_chapter and ch.content:
                    chapter_ready = True
                    break

        return {
            "success": True,
            "has_active_story": True,
            "story": {
                "story_id": story.id,
                "title": story.structure.title if story.structure else story.prompt[:50],
                "current_chapter": current_chapter,
                "chapter_ready": chapter_ready
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching active story: {str(e)}")


# ============================================================================
# Voice Direction (Audiobook Production)
# ============================================================================

@router.post("/stories/{story_id}/chapters/{chapter_number}/voice-direct")
async def voice_direct_chapter(
    story_id: str,
    chapter_number: int,
    force_regenerate: bool = False
):
    """
    Generate/regenerate SSML markup for a chapter (VoiceDirectorAgent).

    NOTE: This generates SSML only, NOT audio. Use /generate-audio for actual TTS.

    SSML is auto-generated when chapters complete, so this endpoint is mainly for:
    - Force-regenerating SSML after prose edits
    - Manual triggering if auto-generation failed

    Cost: ~$0.01 (cheap LLM call)

    The VoiceDirectorAgent (Jim Dale persona) analyzes the chapter and generates:
    - SSML-optimized narration (stored in chapter.tts_content)
    - Character voice mappings
    - Emotional beat annotations
    - Estimated duration

    Args:
        story_id: Story ID
        chapter_number: Chapter to voice direct
        force_regenerate: If True, regenerate even if tts_content exists

    Returns:
        Dict with ssml_length, estimated_duration_seconds (no audio_base64!)
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        result = await _coordinator.voice_direct_chapter(
            story_id=story_id,
            chapter_number=chapter_number,
            force_regenerate=force_regenerate
        )

        if not result.get("success"):
            error_msg = result.get("error", "Unknown error")
            raise HTTPException(status_code=400, detail=error_msg)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice direction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice direction failed: {str(e)}")


@router.post("/stories/{story_id}/chapters/{chapter_number}/generate-audio")
async def generate_chapter_audio(
    story_id: str,
    chapter_number: int,
    force_regenerate: bool = False
):
    """
    Generate or retrieve TTS audio for a chapter.

    Flow:
    1. Check if audio_blob_url exists in chapter (cached audio)
    2. If yes & not force_regenerate: return SAS URL for cached audio
    3. If no: generate TTS, upload to blob storage, save URL, return SAS URL
    4. Fallback to base64 streaming if blob upload fails

    Args:
        story_id: Story ID
        chapter_number: Chapter number (1-indexed)
        force_regenerate: If True, regenerate even if cached audio exists

    Returns:
        Dict with audio_url (SAS URL) or fallback_audio_base64
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    # Use module-level blob_storage_service (set by main.py via set_blob_storage_service)
    blob_storage_service = _blob_storage_service

    try:
        # 1. Get story and find chapter
        story = await _coordinator.storage.get_story(story_id)
        if not story:
            raise HTTPException(status_code=404, detail=f"Story {story_id} not found")

        chapter = None
        for ch in story.chapters or []:
            if ch.number == chapter_number:
                chapter = ch
                break

        if not chapter:
            raise HTTPException(status_code=404, detail=f"Chapter {chapter_number} not found")

        # 2. Check chapter has content
        if not chapter.content:
            raise HTTPException(
                status_code=400,
                detail="Chapter has no content yet."
            )

        # 3. Estimate duration (roughly 150 words per minute for audiobooks)
        word_count = len(chapter.content.split())
        duration_estimate = int((word_count / 150) * 60)  # seconds
        if chapter.voice_direction_metadata:
            duration_estimate = chapter.voice_direction_metadata.total_estimated_duration_seconds

        # 4. Check for cached audio in blob storage
        if chapter.audio_blob_url and not force_regenerate and blob_storage_service:
            logger.info(f"🔊 Found cached audio for Chapter {chapter_number}, generating SAS URL")
            try:
                household_id = await _coordinator.storage.get_household_id_for_story(story_id)
                if household_id:
                    sas_url = await blob_storage_service.get_audio_url(
                        household_id=household_id,
                        story_id=story_id,
                        chapter_num=chapter_number,
                        expiry_hours=24
                    )
                    if sas_url:
                        logger.info(f"✅ Returning cached audio SAS URL for Chapter {chapter_number}")
                        return {
                            "success": True,
                            "audio_url": sas_url,
                            "fallback_audio_base64": None,
                            "source": "blob_cache",
                            "chapter_number": chapter_number,
                            "duration_estimate": duration_estimate,
                            "provider": "cached"
                        }
            except Exception as e:
                logger.warning(f"Failed to get cached audio SAS URL: {e}, will regenerate")

        # 5. Get language from story preferences
        lang_code = story.preferences.language if story.preferences else "en"
        language = get_tts_language_code(lang_code)

        # 6. Determine which content to use:
        # - ElevenLabs: Use tts_content with [audio tags] for expressive narration
        # - Speechify: Use tts_content (SSML tags supported)
        # - Google TTS fallback: Use tts_content (SSML)
        # - Fallback: Use plain content if tts_content not available
        use_ssml = False

        if chapter.tts_content:
            # Use VoiceDirectorAgent output with [emotion] tags
            tts_text = chapter.tts_content
            use_ssml = voice_service.provider == 'google'  # Only Google needs SSML flag
            logger.info(f"🔊 Using voice-directed content ({len(tts_text)} chars) with {voice_service.provider}")
        else:
            # Fallback to plain prose
            tts_text = chapter.content
            logger.info(f"🔊 Using plain prose ({len(tts_text)} chars) - no voice direction available")

        logger.info(f"🔊 Generating audio for Chapter {chapter_number} (lang: {language})")

        # === OBSERVATORY: Emit TTS request started ===
        await story_events.emit("tts_request_started", story_id, {
            "chapter_number": chapter_number,
            "content_length": len(tts_text),
            "provider": voice_service.provider,
            "language": language,
            "use_ssml": use_ssml
        })

        # 7. Generate TTS audio
        audio_bytes = await voice_service.text_to_speech(
            text=tts_text,
            language_code=language,
            use_ssml=use_ssml,
            use_case="narration"
        )

        if not audio_bytes:
            # Update chapter TTS status
            chapter.tts_status = "failed"
            chapter.tts_error = "TTS generation returned no audio"
            await _coordinator.storage.save_chapter(story_id, chapter)

            # === OBSERVATORY: Emit TTS failed ===
            await story_events.emit("tts_failed", story_id, {
                "chapter_number": chapter_number,
                "error": "TTS generation returned no audio",
                "provider": voice_service.provider
            })

            raise HTTPException(status_code=500, detail="TTS generation failed - no audio returned")

        logger.info(f"✅ Audio generated: {len(audio_bytes)} bytes, ~{duration_estimate}s")

        # 8. Try to upload to blob storage
        if blob_storage_service:
            try:
                household_id = await _coordinator.storage.get_household_id_for_story(story_id)
                if household_id:
                    # Upload audio to blob storage
                    blob_url = await blob_storage_service.upload_audio(
                        household_id=household_id,
                        story_id=story_id,
                        chapter_num=chapter_number,
                        audio_data=audio_bytes
                    )

                    if blob_url:
                        # Generate SAS URL for immediate use
                        sas_url = await blob_storage_service.get_audio_url(
                            household_id=household_id,
                            story_id=story_id,
                            chapter_num=chapter_number,
                            expiry_hours=24
                        )

                        # Save blob URL to chapter
                        chapter.audio_blob_url = blob_url
                        chapter.tts_status = "ready"
                        chapter.tts_error = None
                        await _coordinator.storage.save_chapter(story_id, chapter)

                        logger.info(f"✅ Audio uploaded to blob storage: {blob_url}")

                        # === OBSERVATORY: Emit TTS completed (blob storage) ===
                        await story_events.emit("tts_completed", story_id, {
                            "chapter_number": chapter_number,
                            "audio_bytes": len(audio_bytes),
                            "duration_estimate": duration_estimate,
                            "provider": voice_service.provider,
                            "storage": "blob"
                        })

                        return {
                            "success": True,
                            "audio_url": sas_url,
                            "fallback_audio_base64": None,
                            "source": "blob_new",
                            "chapter_number": chapter_number,
                            "duration_estimate": duration_estimate,
                            "provider": voice_service.provider
                        }
            except Exception as e:
                logger.warning(f"Blob upload failed, falling back to base64: {e}")

        # 9. Fallback to base64 streaming (no blob storage or upload failed)
        logger.info(f"📤 Returning audio as base64 (blob storage unavailable)")

        # === OBSERVATORY: Emit TTS completed (base64 fallback) ===
        await story_events.emit("tts_completed", story_id, {
            "chapter_number": chapter_number,
            "audio_bytes": len(audio_bytes),
            "duration_estimate": duration_estimate,
            "provider": voice_service.provider,
            "storage": "base64"
        })

        return {
            "success": True,
            "audio_url": None,
            "fallback_audio_base64": base64.b64encode(audio_bytes).decode('utf-8'),
            "source": "generated_base64",
            "chapter_number": chapter_number,
            "duration_estimate": duration_estimate,
            "provider": voice_service.provider
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")

        # === OBSERVATORY: Emit TTS failed ===
        await story_events.emit("tts_failed", story_id, {
            "chapter_number": chapter_number,
            "error": str(e),
            "provider": voice_service.provider if voice_service else "unknown"
        })

        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")


@router.post("/stories/{story_id}/chapters/{chapter_number}/refine-norwegian")
async def refine_chapter_norwegian(
    story_id: str,
    chapter_number: int
):
    """
    Run Norwegian language refinement on a chapter using Borealis via Ollama.

    This is a debug endpoint to test the Borealis refinement without generating
    a new story.

    Args:
        story_id: Story ID
        chapter_number: Chapter number to refine

    Returns:
        Dict with original and refined content
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        # Get the story
        story = await _coordinator.storage.get_story(story_id)
        if not story:
            raise HTTPException(status_code=404, detail=f"Story {story_id} not found")

        # Get the chapter
        chapter = None
        for ch in story.chapters:
            if ch.number == chapter_number:
                chapter = ch
                break

        if not chapter:
            raise HTTPException(status_code=404, detail=f"Chapter {chapter_number} not found")

        # Check if Norwegian
        language = story.preferences.language if story.preferences else "en"
        if language != "no":
            raise HTTPException(status_code=400, detail=f"Refinement only available for Norwegian stories (current: {language})")

        # Get original content
        original_content = chapter.content
        original_words = len(original_content.split())

        # Emit start event
        await story_events.emit("language_refinement_started", story_id, {
            "chapter": chapter_number,
            "language": language,
            "model": "ollama/hf.co/NbAiLab/borealis-4b-instruct-preview-gguf:BF16",
            "content_length": len(original_content)
        })

        # Run refinement (using private method directly for debug endpoint)
        refined_content = await _coordinator._refine_language(
            story_id=story_id,
            chapter_content=original_content,
            chapter_number=chapter_number,
            language=language
        )

        refined_words = len(refined_content.split())
        was_refined = refined_content != original_content

        if was_refined:
            # Update chapter with refined content
            chapter.content = refined_content
            await _coordinator.storage.save_chapter(story_id, chapter)

            logger.info(f"✅ Norwegian refinement saved for Chapter {chapter_number}: {original_words} → {refined_words} words")

        return {
            "success": True,
            "story_id": story_id,
            "chapter_number": chapter_number,
            "was_refined": was_refined,
            "original_word_count": original_words,
            "refined_word_count": refined_words,
            "word_diff": refined_words - original_words,
            "original_content": original_content[:500] + "..." if len(original_content) > 500 else original_content,
            "refined_content": refined_content[:500] + "..." if len(refined_content) > 500 else refined_content,
            # Full content for comparison UI
            "original_content_full": original_content,
            "refined_content_full": refined_content
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Norwegian refinement failed: {e}")
        await story_events.emit("language_refinement_skipped", story_id, {
            "chapter": chapter_number,
            "language": "no",
            "reason": str(e)[:100]
        })
        raise HTTPException(status_code=500, detail=f"Refinement failed: {str(e)}")


@router.post("/stories/{story_id}/voice-direct-all")
async def voice_direct_all_chapters(
    story_id: str,
    force_regenerate: bool = False
):
    """
    Apply voice direction to all chapters in a story.

    Processes each chapter sequentially through the VoiceDirectorAgent.

    Args:
        story_id: Story ID
        force_regenerate: If True, regenerate even if tts_content exists

    Returns:
        Dict with results for each chapter
    """
    if not _coordinator:
        raise HTTPException(status_code=500, detail="Coordinator not initialized")

    try:
        story = await _coordinator.storage.get_story(story_id)
        if not story:
            raise HTTPException(status_code=404, detail=f"Story {story_id} not found")

        if not story.chapters:
            raise HTTPException(status_code=400, detail="Story has no chapters to voice direct")

        results = []
        for chapter in story.chapters:
            result = await _coordinator.voice_direct_chapter(
                story_id=story_id,
                chapter_number=chapter.number,
                force_regenerate=force_regenerate
            )
            results.append({
                "chapter_number": chapter.number,
                "title": chapter.title,
                **result
            })

        successful = sum(1 for r in results if r.get("success"))

        return {
            "success": True,
            "story_id": story_id,
            "chapters_processed": len(results),
            "chapters_successful": successful,
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice direction batch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice direction batch failed: {str(e)}")


# ============================================================================
# One-Chapter-At-A-Time: Prefetch Event Handler
# ============================================================================

async def _handle_prefetch_chapter(event):
    """
    Handle prefetch_chapter event from CompanionAgent.

    This is triggered when a user starts playing a chapter.
    We generate the next chapter in the background so it's ready
    when the user finishes the current chapter.

    Args:
        event: StoryEvent with data containing chapter_number
    """
    story_id = event.story_id
    data = event.data
    chapter_number = data.get("chapter_number")
    triggered_by = data.get("triggered_by_chapter", chapter_number - 1)

    if not chapter_number:
        logger.error(f"prefetch_chapter event missing chapter_number for {story_id}")
        return

    if not _coordinator:
        logger.error(f"Cannot prefetch chapter - coordinator not initialized")
        return

    logger.info(f"📚 Prefetching Chapter {chapter_number} for {story_id} (triggered by Chapter {triggered_by} playback)")

    try:
        # Emit generating event
        await story_events.emit("chapter_generating", story_id, {
            "chapter_number": chapter_number,
            "message": f"Preparing Chapter {chapter_number}...",
            "prefetch": True
        })

        # Generate the chapter
        chapter_result = await _coordinator.write_chapter(story_id, chapter_number)

        if chapter_result["success"]:
            logger.info(f"✅ Prefetch complete: Chapter {chapter_number} for {story_id}")

            # Notify companion agent that prefetch is complete
            companion = get_companion_agent()
            if companion:
                await companion.on_chapter_prefetch_complete(story_id, chapter_number)

            # NOTE: chapter_ready is already emitted by write_chapter() - don't duplicate here
        else:
            error_msg = chapter_result.get("error", "Unknown error")
            logger.error(f"❌ Prefetch failed: Chapter {chapter_number} for {story_id}: {error_msg}")

            await story_events.emit("error", story_id, {
                "stage": "chapter_prefetch",
                "chapter_number": chapter_number,
                "error": error_msg,
                "message": f"Failed to prepare Chapter {chapter_number}. It will be generated when needed."
            })

    except Exception as e:
        logger.error(f"❌ Prefetch exception: Chapter {chapter_number} for {story_id}: {e}")
        await story_events.emit("error", story_id, {
            "stage": "chapter_prefetch",
            "chapter_number": chapter_number,
            "error": str(e),
            "message": f"Error preparing Chapter {chapter_number}."
        })


def setup_prefetch_listener():
    """
    Subscribe to prefetch_chapter events.

    Call this on app startup after story_events is ready.
    """
    story_events.on("prefetch_chapter", _handle_prefetch_chapter)
    logger.info("📚 Prefetch chapter listener registered")
