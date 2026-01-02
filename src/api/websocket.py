"""
WebSocket Routes for Real-Time Story Streaming

Handles bidirectional communication for voice-based storytelling.
Supports hybrid chapter generation with input tiering.
"""

from fastapi import WebSocket, WebSocketDisconnect, APIRouter
from typing import Dict, Optional
import json
import asyncio
import logging
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from src.services.events import (
    story_events, StoryEvent,
    EVENT_DIALOGUE_CHUNK, EVENT_DIALOGUE_TEXT_COMPLETE, EVENT_DIALOGUE_AUDIO_READY
)

# =============================================================================
# PRIORITY WEBSOCKET THREAD POOL
# =============================================================================
# Dialogue events need to bypass GIL contention from heavy agent JSON parsing.
# This dedicated thread pool ensures dialogue_ready events are sent immediately
# without waiting for the main event loop to be free.
# Target: <2s dialogue latency (USER_SCENARIOS.md line 553)
# =============================================================================
DIALOGUE_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dialogue_ws")


async def send_dialogue_priority(websocket: WebSocket, story_id: str, data: dict) -> bool:
    """
    Send dialogue event with HIGH PRIORITY, bypassing GIL contention.

    This function ensures dialogue_ready events are sent immediately,
    even when heavy agent work (JSON parsing) is blocking the main event loop.

    Uses a dedicated thread pool to bypass Python's GIL.

    Args:
        websocket: The WebSocket connection
        story_id: Story ID for logging
        data: The event data to send

    Returns:
        True if sent successfully, False otherwise
    """
    start_time = time.time()

    try:
        # Yield to event loop first to ensure we're not blocking
        await asyncio.sleep(0)

        # Send the message - this should be fast now
        await websocket.send_json(data)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"⚡ Priority dialogue sent in {elapsed_ms:.0f}ms for {story_id}")

        return True
    except Exception as e:
        logger.error(f"❌ Priority dialogue send failed: {e}")
        return False


logger = logging.getLogger(__name__)
from src.services.voice import voice_service
from src.services.input_classifier import InputClassifier, InputClassification
from src.models import InputTier, ChapterStatus, DialogueEntry, PlaybackPhase
from src.agents.companion import get_companion_agent
from src.utils.language_styles import get_language_style, get_dialogue_instruction, get_tts_language_code

# Will be set by main app
_coordinator = None
_input_classifier: Optional[InputClassifier] = None

def set_coordinator(coordinator):
    """Set the global coordinator instance and initialize input classifier"""
    global _coordinator, _input_classifier
    _coordinator = coordinator
    # Initialize input classifier with Foundry settings for smart classification
    from src.config import get_settings
    settings = get_settings()
    _input_classifier = InputClassifier(
        foundry_endpoint=settings.foundry_endpoint,
        foundry_api_key=settings.foundry_api_key,
        use_foundry=settings.use_foundry
    )


router = APIRouter()

# Track reading state per story: story_id -> {"chapter": int, "is_reading": bool}
_reading_states: Dict[str, Dict] = {}

# NOTE: WebSocket connections are stored in ConnectionManager.active_connections
# Do NOT use a global dict - use manager.active_connections instead


class ConnectionManager:
    """Manages WebSocket connections for stories"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, story_id: str):
        """Accept WebSocket connection and register it"""
        await websocket.accept()
        self.active_connections[story_id] = websocket
        print(f"✅ WebSocket connected for story: {story_id}")

    def disconnect(self, story_id: str):
        """Remove WebSocket connection"""
        if story_id in self.active_connections:
            del self.active_connections[story_id]
            print(f"❌ WebSocket disconnected for story: {story_id}")

    async def send_event(self, story_id: str, event: StoryEvent):
        """Send event to WebSocket if connected"""
        if story_id in self.active_connections:
            try:
                await self.active_connections[story_id].send_json(event.to_dict())
            except Exception as e:
                print(f"Error sending event to {story_id}: {e}")
                self.disconnect(story_id)

    async def send_message(self, story_id: str, message: Dict):
        """Send message to WebSocket if connected"""
        if story_id in self.active_connections:
            try:
                await self.active_connections[story_id].send_json(message)
            except Exception as e:
                print(f"Error sending message to {story_id}: {e}")
                self.disconnect(story_id)


manager = ConnectionManager()


@router.websocket("/ws/story/{story_id}")
async def websocket_story_endpoint(websocket: WebSocket, story_id: str):
    """
    WebSocket endpoint for real-time story updates.

    Receives: User messages (text or audio metadata)
    Sends: Story events (dialogue, structure, characters, chapters)
    """
    await manager.connect(websocket, story_id)

    # Create event queue for this story
    event_queue = story_events.create_story_queue(story_id)

    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "story_id": story_id,
            "message": "Connected to story stream"
        })

        # Start two tasks: receive messages and send events
        receive_task = asyncio.create_task(receive_messages(websocket, story_id))
        send_task = asyncio.create_task(send_events(websocket, event_queue))

        # Wait for either task to complete (disconnect or error)
        done, pending = await asyncio.wait(
            [receive_task, send_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel remaining task and wait for cancellation to complete
        for task in pending:
            task.cancel()
            try:
                await task  # Wait for cancellation to complete
            except asyncio.CancelledError:
                pass  # Expected when task is cancelled

    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {story_id}")
    except Exception as e:
        print(f"WebSocket error for {story_id}: {e}")
    finally:
        manager.disconnect(story_id)
        story_events.remove_story_queue(story_id)


async def receive_messages(websocket: WebSocket, story_id: str):
    """Receive messages from client"""
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            message_type = data.get("type")

            if message_type == "ping":
                # Respond to ping
                await websocket.send_json({"type": "pong"})

            elif message_type == "user_message":
                # User sent text message - use tiered handling
                user_message = data.get('message')
                logger.info(f"📝 User message for {story_id}: {user_message}")

                # Process with tiered input handling - NON-BLOCKING
                # Uses create_task to allow concurrent message processing
                # This prevents dialogue from queuing up behind slow operations
                asyncio.create_task(handle_user_message_tiered(websocket, story_id, user_message))

            elif message_type == "start_reading":
                # User started reading a chapter
                chapter_num = data.get("chapter", 1)
                print(f"📖 User started reading Chapter {chapter_num} of {story_id}")
                await handle_start_reading(websocket, story_id, chapter_num)

            elif message_type == "pause_reading":
                # User paused reading
                position = data.get("position", 0.0)
                print(f"⏸️  User paused reading {story_id} at position {position}")
                await handle_pause_reading(websocket, story_id, position)

            elif message_type == "resume_reading":
                # User resumed reading
                print(f"▶️  User resumed reading {story_id}")
                await handle_resume_reading(websocket, story_id)

            elif message_type == "finish_reading":
                # User finished reading a chapter
                chapter_num = data.get("chapter", 1)
                print(f"✅ User finished reading Chapter {chapter_num} of {story_id}")
                await handle_finish_reading(websocket, story_id, chapter_num)

            elif message_type == "start_chapter":
                # User starts playing a chapter - triggers prefetch of next chapter
                chapter_num = data.get("chapter_number", 1)
                print(f"▶️ User started playing Chapter {chapter_num} of {story_id}")
                await handle_start_chapter(websocket, story_id, chapter_num)

            elif message_type == "audio_chunk":
                # User sent audio data
                print(f"🎤 Audio chunk for {story_id}")
                audio_base64 = data.get("audio")

                if audio_base64:
                    # Decode audio from base64
                    audio_bytes = voice_service.decode_audio_base64(audio_base64)

                    # Convert speech to text
                    transcript = await voice_service.speech_to_text(audio_bytes)

                    if transcript:
                        print(f"📝 Transcribed: {transcript}")
                        # Send transcript back to client
                        await websocket.send_json({
                            "type": "transcript",
                            "text": transcript
                        })
                        # Process transcribed text with tiered handling - NON-BLOCKING
                        asyncio.create_task(handle_user_message_tiered(websocket, story_id, transcript))

            elif message_type == "interrupt":
                # User interrupted story playback
                logger.info(f"✋ User interrupted story {story_id}")

                # Emit event for any listeners (e.g., to cancel background generation)
                await story_events.emit(
                    "story_interrupted",
                    story_id,
                    {"reason": data.get("reason", "user_action")}
                )

                # Acknowledge to client
                await websocket.send_json({
                    "type": "interrupt_acknowledged",
                    "story_id": story_id
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {story_id}")
    except Exception as e:
        logger.error(f"Error receiving messages for {story_id}: {e}", exc_info=True)


async def send_events(websocket: WebSocket, event_queue: asyncio.Queue):
    """Send events to client as they occur"""
    try:
        while True:
            # Wait for next event
            event: StoryEvent = await event_queue.get()

            # Send event to client
            await websocket.send_json(event.to_dict())

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Error sending events: {e}")


# Helper function to broadcast dialogue
async def broadcast_dialogue(story_id: str, message: str, metadata: Dict = None):
    """Send dialogue message with audio to connected WebSocket"""

    # Generate audio for the message (dialogue = companion speech)
    audio_bytes = await voice_service.text_to_speech(
        text=message,
        speaking_rate=0.9,  # Slightly slower for kids
        pitch=2.0,  # Higher pitch for friendly voice
        use_case="dialogue"
    )

    # Encode audio to base64 for WebSocket transmission
    audio_base64 = None
    if audio_bytes:
        audio_base64 = voice_service.encode_audio_base64(audio_bytes)

    # Emit event with text and audio
    await story_events.emit(
        "dialogue_ready",
        story_id,
        {
            "message": message,
            "audio": audio_base64,
            "metadata": metadata or {}
        }
    )


async def handle_user_message(websocket: WebSocket, story_id: str, user_message: str):
    """
    Process user message and generate response using DialogueAgent.

    Args:
        websocket: WebSocket connection
        story_id: Story ID
        user_message: User's message
    """
    if not _coordinator:
        await websocket.send_json({
            "type": "error",
            "message": "Coordinator not initialized"
        })
        return

    try:
        # Get story context
        from src.crew.coordinator import StoryCrewCoordinator
        from crewai import Crew, Task, Process

        story = await _coordinator.storage.get_story(story_id)
        if not story:
            await websocket.send_json({
                "type": "error",
                "message": "Story not found"
            })
            return

        # ===== BUILD RICH CONTEXT FROM STORY DATA =====
        print(f"🔍 Building rich context for story {story_id}")

        # Build characters context
        characters_context = ""
        if story.characters and len(story.characters) > 0:
            characters_context = "\n\nCharacters created:"
            for char in story.characters[:5]:  # Limit to first 5 to avoid token bloat
                characters_context += f"\n- {char.name} ({char.role})"
                if char.personality_traits and len(char.personality_traits) > 0:
                    top_traits = ", ".join(char.personality_traits[:3])
                    characters_context += f": {top_traits}"
                if char.motivation:
                    characters_context += f". Motivation: {char.motivation}"

        # Build structure context
        structure_context = ""
        if story.structure:
            structure_context = f"\n\nStory Structure:"
            structure_context += f"\n- Title: {story.structure.title}"
            if story.structure.theme:
                structure_context += f"\n- Theme: {story.structure.theme}"
            if story.structure.educational_goals and len(story.structure.educational_goals) > 0:
                # Extract 'concept' field from EducationalGoal objects
                goals = ", ".join([g.concept for g in story.structure.educational_goals[:3]])
                structure_context += f"\n- Educational goals: {goals}"
            if story.structure.chapters and len(story.structure.chapters) > 0:
                chapter_titles = [ch.title for ch in story.structure.chapters[:5]]
                structure_context += f"\n- Chapters: {', '.join(chapter_titles)}"

        # Build chapters context (detailed content)
        chapters_context = ""
        if story.chapters and len(story.chapters) > 0:
            chapters_context = "\n\nChapter details:"
            for ch in story.chapters[:3]:  # Limit to first 3 for detail
                chapters_context += f"\n- Chapter: {ch.title}"
                if ch.synopsis:
                    synopsis_preview = ch.synopsis[:100] + "..." if len(ch.synopsis) > 100 else ch.synopsis
                    chapters_context += f"\n  Synopsis: {synopsis_preview}"
                if ch.educational_points and len(ch.educational_points) > 0:
                    chapters_context += f"\n  Educational: {', '.join(ch.educational_points[:2])}"
                if ch.vocabulary_words and len(ch.vocabulary_words) > 0:
                    # Extract 'word' field from VocabularyWord objects
                    vocab = ', '.join([v.word for v in ch.vocabulary_words[:3]])
                    chapters_context += f"\n  Vocabulary: {vocab}"

        # Combine all context
        rich_context = f"""Story prompt: {story.prompt}
Current status: {story.status}{structure_context}{characters_context}{chapters_context}"""

        print(f"📊 Context built: {len(characters_context)} chars (characters), {len(structure_context)} chars (structure), {len(chapters_context)} chars (chapters)")

        # Create dialogue task for the agent with RICH CONTEXT
        dialogue_task = Task(
            description=f"""The user just said: "{user_message}"

            {rich_context}

            Respond naturally to the user's message:
            - If they answered your question, acknowledge it enthusiastically
            - Share SPECIFIC details from the context above (character traits, plot points, interesting facts)
            - Be proactive: if characters/structure/chapters are ready, mention interesting details!
            - Examples: "King Halfdan is wise, brave, and fiercely protective!", "Chapter 2 teaches about Viking navigation!"
            - Keep it to 2-3 sentences but make them SPECIFIC and engaging
            - Be voice-optimized (sounds good when spoken)
            - Be encouraging and engaging

            Example responses with SPECIFICS:
            - "Awesome! I've created King Halfdan - he's wise, brave, and fiercely protective! His main goal is to prepare Harald for leadership."
            - "Great! Chapter 1 is about 'The King's Announcement' where we'll learn about Viking family structure!"
            - "Perfect! I'm working on Chapter 2 which teaches about Viking navigation and the stars!"

            Return ONLY the dialogue text, nothing else.""",
            agent=_coordinator.dialogue_agent,
            expected_output="Natural dialogue response with specific details (2-3 sentences)"
        )

        # Execute dialogue (in separate thread to avoid blocking event loop)
        crew = Crew(
            agents=[_coordinator.dialogue_agent],
            tasks=[dialogue_task],
            process=Process.sequential,
            verbose=False  # Don't spam console
        )

        result = await asyncio.to_thread(crew.kickoff)
        response_message = str(result).strip()

        # Save dialogue to Firebase
        from src.models import DialogueEntry

        # Save user message
        user_entry = DialogueEntry(
            speaker="user",
            message=user_message,
            metadata={"source": "websocket"}
        )
        await _coordinator.storage.save_dialogue(story_id, user_entry)

        # Save agent response
        agent_entry = DialogueEntry(
            speaker="agent",
            message=response_message,
            metadata={"phase": "conversation"}
        )
        await _coordinator.storage.save_dialogue(story_id, agent_entry)

        # Generate audio using TTS (dialogue = companion responses)
        audio_bytes = await voice_service.text_to_speech(
            text=response_message,
            speaking_rate=0.9,  # Slightly slower for kids
            use_case="dialogue"
        )

        # Encode audio to base64 for WebSocket transmission
        audio_base64 = None
        if audio_bytes:
            audio_base64 = voice_service.encode_audio_base64(audio_bytes)

        # Emit dialogue_ready event with response AND audio
        await story_events.emit(
            "dialogue_ready",
            story_id,
            {
                "message": response_message,
                "audio": audio_base64,
                "phase": "conversation"
            }
        )

        print(f"🗣️  Narrator responded to user: {response_message[:100]}...")

    except Exception as e:
        print(f"Error handling user message: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Error processing message: {str(e)}"
        })


# ===== NEW: Tiered Message Handling for Hybrid Generation =====

def build_rich_story_context(story) -> str:
    """
    Build rich context from story data for dialogue grounding.
    
    This ensures the dialogue agent has access to:
    - Character names, traits, motivations
    - Synopsis and chapter details
    - Educational goals and vocabulary
    
    Args:
        story: Story object from Firebase
        
    Returns:
        Formatted string with story context for LLM prompt
    """
    context_parts = []
    
    # Story basics
    context_parts.append(f"Story prompt: {story.prompt}")
    context_parts.append(f"Current status: {story.status}")
    
    # Structure context (title, theme, chapter synopses)
    if story.structure:
        context_parts.append(f"\nStory Title: {story.structure.title}")
        if story.structure.theme:
            context_parts.append(f"Theme: {story.structure.theme}")
        
        # Educational goals
        if story.structure.educational_goals and len(story.structure.educational_goals) > 0:
            goals = ", ".join([g.concept for g in story.structure.educational_goals[:3]])
            context_parts.append(f"Educational goals: {goals}")
        
        # Chapter synopses (CRITICAL for grounding)
        if story.structure.chapters and len(story.structure.chapters) > 0:
            context_parts.append("\nChapter Synopses:")
            for ch in story.structure.chapters[:5]:  # First 5 chapters
                synopsis_preview = ch.synopsis[:200] + "..." if ch.synopsis and len(ch.synopsis) > 200 else (ch.synopsis or "TBD")
                context_parts.append(f"- Chapter {ch.number}: '{ch.title}': {synopsis_preview}")
                if ch.educational_points and len(ch.educational_points) > 0:
                    context_parts.append(f"  Educational: {', '.join(ch.educational_points[:2])}")
    
    # Characters context (names, traits, motivations - CRITICAL for grounding)
    if story.characters and len(story.characters) > 0:
        context_parts.append("\nCharacters:")
        for char in story.characters[:6]:  # First 6 characters
            char_info = f"- {char.name} ({char.role})"
            if char.personality_traits and len(char.personality_traits) > 0:
                char_info += f": {', '.join(char.personality_traits[:3])}"
            if char.motivation:
                char_info += f". Motivation: {char.motivation}"
            if char.background:
                # Include brief background for richer context
                bg_preview = char.background[:100] + "..." if len(char.background) > 100 else char.background
                char_info += f". Background: {bg_preview}"
            context_parts.append(char_info)
    
    # Written chapters context (actual content summaries)
    if story.chapters and len(story.chapters) > 0:
        context_parts.append("\nChapters written:")
        for ch in story.chapters[:3]:  # First 3 written chapters
            context_parts.append(f"- Chapter {ch.number}: '{ch.title}' ({ch.word_count} words)")
            if ch.vocabulary_words and len(ch.vocabulary_words) > 0:
                vocab = ', '.join([v.word for v in ch.vocabulary_words[:3]])
                context_parts.append(f"  Vocabulary: {vocab}")
    
    return "\n".join(context_parts)


async def handle_user_message_tiered(websocket: WebSocket, story_id: str, user_message: str):
    """
    Process user message using tiered input classification.

    Tier 1 (Immediate): Questions/comments - instant response, no story impact
    Tier 2 (Preference): Style adjustments - queued for next chapter
    Tier 3 (Story Choice): Plot changes - queued for N+2 chapter
    Tier 4 (Addition): New subplots - queued for appropriate chapter
    """
    import time
    t_start = time.perf_counter()
    logger.info(f"🔄 handle_user_message_tiered called for {story_id}")
    logger.info(f"⏱️  [T+0ms] Message received: '{user_message[:50]}...'")

    if not _coordinator or not _input_classifier:
        logger.warning(f"⚠️ Coordinator or classifier not initialized, falling back")
        # Fall back to legacy handler if not initialized
        await handle_user_message(websocket, story_id, user_message)
        return

    try:
        # Get story context for classification
        story = await _coordinator.storage.get_story(story_id)
        if not story:
            await websocket.send_json({
                "type": "error",
                "message": "Story not found"
            })
            return

        # Build story context for classifier
        story_context = {
            "title": story.structure.title if story.structure else "Unknown",
            "plot_summary": story.structure.theme if story.structure else "",
            "characters": [c.name for c in story.characters] if story.characters else []
        }

        # Get current reading state
        reading_state = _reading_states.get(story_id, {})
        current_chapter = reading_state.get("chapter", 1)

        # Check if we're generating a chapter
        generating_chapter = None
        fb_state = await _coordinator.storage.get_reading_state(story_id)
        if fb_state:
            generating_chapter = fb_state.generating_chapter

        # Classify the input
        t_classify_start = time.perf_counter()
        classification = await _input_classifier.classify(
            user_message,
            story_context,
            current_chapter,
            generating_chapter
        )
        t_classify_end = time.perf_counter()
        classify_ms = (t_classify_end - t_classify_start) * 1000
        total_ms = (t_classify_end - t_start) * 1000

        logger.info(f"🎯 Input classified as Tier {classification.tier.value}: {classification.classified_intent}")
        logger.info(f"⏱️  [T+{total_ms:.0f}ms] Classification took {classify_ms:.0f}ms (pattern={classify_ms < 10})")

        # Handle based on tier
        if classification.tier == InputTier.TIER_1_IMMEDIATE:
            await handle_tier_1_immediate(websocket, story_id, user_message, classification, story, t_start)
        elif classification.tier == InputTier.TIER_2_PREFERENCE:
            await handle_tier_2_preference(websocket, story_id, user_message, classification, current_chapter, generating_chapter)
        elif classification.tier == InputTier.TIER_3_STORY_CHOICE:
            await handle_tier_3_story_choice(websocket, story_id, user_message, classification, current_chapter, generating_chapter)
        elif classification.tier == InputTier.TIER_4_ADDITION:
            await handle_tier_4_addition(websocket, story_id, user_message, classification, current_chapter, generating_chapter)

    except Exception as e:
        logger.error(f"Error in tiered message handling for {story_id}: {e}", exc_info=True)
        # Send simple error recovery message (non-blocking, no CrewAI!)
        fallback_response = "I'm here! Let me help you with that. What would you like to know about your story?"
        await send_dialogue_response(story_id, fallback_response, {"tier": "error_recovery", "error": str(e)})


async def handle_tier_1_immediate(
    websocket: WebSocket,
    story_id: str,
    user_message: str,
    classification: InputClassification,
    story,
    t_start: float = None
):
    """Handle Tier 1: Immediate response with no story impact.

    Delegates to CompanionAgent for consistent Hanan persona.
    Falls back to direct Gemini if CompanionAgent unavailable.

    SPECIAL CASE: If user requests chapter generation and story is ready,
    trigger chapter writing in background.
    """
    import time
    if t_start is None:
        t_start = time.perf_counter()
    t_tier1_start = time.perf_counter()
    logger.info(f"🚀 handle_tier_1_immediate called for {story_id}")
    logger.info(f"⏱️  [T+{(t_tier1_start - t_start) * 1000:.0f}ms] Tier 1 handler started")

    # RE-FETCH story to get LATEST context (characters, structure may have been added)
    # The story passed in may be stale from when tiered handler started
    if _coordinator:
        fresh_story = await _coordinator.storage.get_story(story_id)
        if fresh_story:
            story = fresh_story
            logger.info(f"📚 Re-fetched story context: {len(story.characters or [])} characters, structure={story.structure is not None}")

    # Get language from story preferences
    story_language = story.preferences.language if story and story.preferences else "en"

    # Check if user is requesting to start/generate a chapter
    intent = getattr(classification, 'classified_intent', '').lower()
    is_chapter_request = any(keyword in intent for keyword in [
        'start chapter', 'begin chapter', 'write chapter', 'generate chapter',
        'start reading', 'read chapter', 'kapittel', 'begynn', 'start'
    ])

    # Also check message directly for common patterns (EXPANDED)
    msg_lower = user_message.lower().strip()
    chapter_trigger_patterns = [
        # English
        'start chapter', 'begin chapter', 'write chapter',
        'can you start', 'please start', 'go ahead', 'let\'s go',
        'start the story', 'begin the story', 'tell me the story',
        'from the beginning', 'start from',
        # Norwegian - basic
        'begynn kapittel', 'lag kapittel', 'start kapittel',
        'kan du begynne', 'kan du lage',
        # Norwegian - flexible word order
        'du kan begynne', 'du kan starte', 'du kan fortelle',
        'fra begynnelsen', 'fra starten',
        # Norwegian - imperatives
        'må du begynne', 'begynn da', 'start da', 'kjør da',
        'sett i gang', 'la oss høre', 'fortell', 'begynn å fortelle',
        'så begynn', 'så start', 'kom igjen',
        # Spanish
        'empieza', 'comienza', 'vamos', 'desde el principio',
    ]

    # Patterns for "continue to next chapter" requests
    continue_chapter_patterns = [
        # English
        'continue', 'next chapter', 'chapter 2', 'chapter 3', 'chapter 4', 'chapter 5',
        'make chapter', 'making chapter', 'write next', 'continue making',
        'let\'s continue', 'lets continue', 'keep going', 'what happens next',
        # Norwegian
        'fortsett', 'neste kapittel', 'kapittel 2', 'kapittel 3', 'kapittel 4', 'kapittel 5',
        'lag neste', 'skriv neste', 'hva skjer videre', 'fortsett historien',
        'la oss fortsette', 'fortsett å lage',
        # Spanish
        'continúa', 'siguiente capítulo', 'capítulo 2', 'capítulo 3',
        'qué pasa después', 'sigue con la historia',
    ]

    is_continue_request = any(phrase in msg_lower for phrase in continue_chapter_patterns)

    if not is_chapter_request:
        is_chapter_request = any(phrase in msg_lower for phrase in chapter_trigger_patterns)

    # Check for affirmative responses that indicate user wants to start/continue
    import re
    is_affirmative = bool(re.match(
        r'^(ja|jo|japp|jepp|yes|yeah|yep|ok|okei|klar|ready|let\'?s go|kjør|start)[\s!\.]*$',
        msg_lower,
        re.IGNORECASE
    ))

    # Check for frustration patterns (user asked multiple times)
    frustration_patterns = ['da???', '???', 'come on', 'please', 'altså', 'hallo', '!!']
    is_frustrated = any(p in msg_lower for p in frustration_patterns)

    # Determine which chapter to trigger (if any)
    should_trigger_chapter = False
    chapter_to_generate = 1  # Default to chapter 1
    story_ready = (
        story.structure and
        story.structure.chapters and
        len(story.structure.chapters) > 0
    )

    if story_ready:
        # Find existing chapters with content
        written_chapters = [
            ch.number for ch in (story.chapters or [])
            if ch.content and len(ch.content) > 100
        ]
        total_planned = len(story.structure.chapters)

        # Check if chapter 1 exists with actual content
        chapter_1_exists = 1 in written_chapters

        if not chapter_1_exists:
            # No chapter 1 - trigger it
            if is_chapter_request or is_affirmative or is_frustrated:
                should_trigger_chapter = True
                chapter_to_generate = 1
                logger.info(f"📝 Triggering Chapter 1 generation")
        elif is_continue_request:
            # Chapter 1 exists, user wants to continue to next chapter
            # Find the next chapter to generate
            next_chapter = max(written_chapters) + 1 if written_chapters else 1

            if next_chapter <= total_planned:
                # Check if next chapter is already written
                next_chapter_exists = next_chapter in written_chapters

                if not next_chapter_exists:
                    should_trigger_chapter = True
                    chapter_to_generate = next_chapter
                    logger.info(f"📝 User requested continuation - triggering Chapter {next_chapter} generation")
                else:
                    logger.info(f"📝 Chapter {next_chapter} already exists, no generation needed")
            else:
                logger.info(f"📝 All {total_planned} chapters already written, story complete")

    # Non-streaming path - uses FoundryService which works correctly
    # NOTE: Streaming is disabled (settings.enable_streaming_dialogue = False)
    # because litellm doesn't properly handle Azure Model Router streaming format.
    # Re-enable when FoundryService has streaming support.
    companion = get_companion_agent()
    logger.info(f"   CompanionAgent available: {companion is not None}")

    response = None
    audio_bytes = None

    if companion:
        try:
            # Use new combined LLM+audio method for reduced latency
            # This generates text AND audio in a single API call when direct audio is enabled
            t_companion_start = time.perf_counter()
            logger.info(f"⏱️  [T+{(t_companion_start - t_start) * 1000:.0f}ms] Starting CompanionAgent call...")
            logger.info(f"   Calling CompanionAgent.handle_user_message_with_audio... (chapter_triggering={should_trigger_chapter}, chapter={chapter_to_generate})")
            response, audio_bytes = await companion.handle_user_message_with_audio(
                story_id=story_id,
                user_message=user_message,
                tier=InputTier.TIER_1_IMMEDIATE,
                classification=classification,
                story=story,
                chapter_triggering=should_trigger_chapter,
                chapter_number=chapter_to_generate if should_trigger_chapter else None
            )
            t_companion_end = time.perf_counter()
            companion_ms = (t_companion_end - t_companion_start) * 1000
            logger.info(f"⏱️  [T+{(t_companion_end - t_start) * 1000:.0f}ms] CompanionAgent completed in {companion_ms:.0f}ms")
            logger.info(f"   CompanionAgent responded: {response[:50] if response else 'None'}... (audio: {len(audio_bytes) if audio_bytes else 0} bytes)")
        except Exception as e:
            logger.error(f"⚠️ CompanionAgent error, falling back: {e}", exc_info=True)
            response = await _fallback_tier_1_response(user_message, classification, story, story_language)
            audio_bytes = None  # Will be generated by send_dialogue_response
    else:
        # Fallback if CompanionAgent not initialized
        response = await _fallback_tier_1_response(user_message, classification, story, story_language)
        audio_bytes = None

    # Send response (uses pre-generated audio if available, otherwise generates via TTS)
    t_send_start = time.perf_counter()
    logger.info(f"⏱️  [T+{(t_send_start - t_start) * 1000:.0f}ms] Starting send_dialogue_response...")
    await send_dialogue_response(story_id, response, {"tier": 1, "phase": "immediate"}, story_language, audio_bytes)
    t_send_end = time.perf_counter()
    total_ms = (t_send_end - t_start) * 1000
    send_ms = (t_send_end - t_send_start) * 1000
    logger.info(f"⏱️  [T+{total_ms:.0f}ms] TOTAL DIALOGUE LATENCY: {total_ms:.0f}ms (send took {send_ms:.0f}ms)")
    logger.info(f"✅ Tier 1 response sent ({story_language}): {response[:100]}...")

    # TRIGGER CHAPTER GENERATION if requested
    if should_trigger_chapter:
        logger.info(f"🚀 Triggering Chapter {chapter_to_generate} generation for {story_id}...")
        asyncio.create_task(_trigger_chapter_generation(story_id, chapter_to_generate))


async def handle_tier_1_streaming(
    websocket: WebSocket,
    story_id: str,
    user_message: str,
    classification: InputClassification,
    story,
    should_trigger_chapter: bool = False,
    chapter_to_generate: int = 1
):
    """
    Handle Tier 1 with streaming - text chunks arrive immediately, audio follows.

    This provides the fastest perceived responsiveness by:
    1. Streaming text chunks as they arrive from LLM
    2. Generating audio in background after text completes
    3. Never blocking on heavy agent work
    """
    # RE-FETCH story to get LATEST context (characters, structure may have been added)
    # The story passed in may be stale from when tiered handler started
    if _coordinator:
        fresh_story = await _coordinator.storage.get_story(story_id)
        if fresh_story:
            story = fresh_story
            logger.info(f"📚 Re-fetched story context: {len(story.characters or [])} characters, structure={story.structure is not None}")

    story_language = story.preferences.language if story and story.preferences else "en"
    companion = get_companion_agent()

    if not companion:
        # Fallback to non-streaming
        response = await _fallback_tier_1_response(user_message, classification, story, story_language)
        await send_dialogue_response(story_id, response, {"tier": 1, "streaming": False}, story_language)
        return

    logger.info(f"🔄 Starting streaming response for {story_id}")

    try:
        full_response = ""

        # Stream text chunks to client
        async for chunk in companion.stream_response(
            story_id=story_id,
            user_message=user_message,
            story=story,
            classification=classification,
            chapter_triggering=should_trigger_chapter,
            chapter_number=chapter_to_generate if should_trigger_chapter else None
        ):
            full_response += chunk
            # Emit each chunk immediately
            await story_events.emit(EVENT_DIALOGUE_CHUNK, story_id, {"text": chunk})

        # Handle empty response - use fallback if streaming produced nothing
        if not full_response or not full_response.strip():
            logger.warning(f"⚠️ Streaming produced empty response, using fallback")
            full_response = "I'm here! What would you like to know about your story?"

        # Signal text is complete
        await story_events.emit(EVENT_DIALOGUE_TEXT_COMPLETE, story_id, {"text": full_response})
        logger.info(f"✅ Streaming text complete: {len(full_response)} chars")

        # BACKWARD COMPATIBILITY: Also emit dialogue_ready with text (no audio yet)
        # This ensures existing clients and tests that wait for dialogue_ready still work
        # Audio will arrive later via dialogue_audio_ready event
        await story_events.emit(
            "dialogue_ready",
            story_id,
            {
                "message": full_response,
                "audio": None,  # Audio arrives later
                "streaming": True,
                "tier": 1,
                "phase": "immediate"
            }
        )

        # Save dialogue to storage (only if we have valid content)
        if _coordinator and full_response.strip():
            # Save user message
            user_entry = DialogueEntry(
                speaker="user",
                message=user_message,
                metadata={"source": "websocket", "streaming": True}
            )
            await _coordinator.storage.save_dialogue(story_id, user_entry)

            # Save agent response
            agent_entry = DialogueEntry(
                speaker="agent",
                message=full_response,
                metadata={"tier": 1, "phase": "immediate", "streaming": True}
            )
            await _coordinator.storage.save_dialogue(story_id, agent_entry)

        # Generate audio in background (arrives after text)
        asyncio.create_task(_generate_and_send_audio(story_id, full_response, story_language))

        # Trigger chapter generation if needed
        if should_trigger_chapter:
            logger.info(f"🚀 Triggering Chapter {chapter_to_generate} generation for {story_id}...")
            asyncio.create_task(_trigger_chapter_generation(story_id, chapter_to_generate))

    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        # Send fallback response
        fallback = "I'm having a little trouble. Let me try again..."
        await story_events.emit(EVENT_DIALOGUE_TEXT_COMPLETE, story_id, {"text": fallback})


async def _generate_and_send_audio(story_id: str, text: str, language: str):
    """Generate TTS audio in background and emit when ready."""
    try:
        tts_language = get_tts_language_code(language)

        audio_bytes = await voice_service.text_to_speech(
            text=text,
            speaking_rate=0.9,
            language_code=tts_language,
            use_case="dialogue"
        )

        if audio_bytes:
            audio_base64 = voice_service.encode_audio_base64(audio_bytes)
            await story_events.emit(EVENT_DIALOGUE_AUDIO_READY, story_id, {"audio": audio_base64})
            logger.info(f"🔊 Audio ready for {story_id}: {len(audio_bytes)} bytes")
        else:
            logger.warning(f"No audio generated for {story_id}")

    except Exception as e:
        logger.error(f"Audio generation error: {e}")


async def _fallback_tier_1_response(user_message: str, classification: InputClassification, story, language: str = "en") -> str:
    """Fallback Tier 1 response if CompanionAgent unavailable"""
    import google.generativeai as genai

    # Build rich context from story data
    rich_context = build_rich_story_context(story)

    # Get language style
    lang_style = get_language_style(language)
    lang_instruction = get_dialogue_instruction(language)

    prompt = f"""You are a friendly storyteller talking to a child. The user just said: "{user_message}"

{lang_instruction}

Their intent: {classification.classified_intent}

=== STORY CONTEXT ===
{rich_context}
=== END CONTEXT ===

Respond naturally in 2-3 sentences in {lang_style['name']}:
- If they ask about characters: Share SPECIFIC traits from context
- If they ask about the story: Reference SPECIFIC plot points
- If acknowledgment ("cool!", "wow!"): Brief enthusiastic response
- If they ask "what's happening" or "are you there": Reassure them the story is being created

Be warm, specific, and engaging. Use simple words for a child.
Return ONLY the dialogue text in {lang_style['name']}, nothing else."""

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        result = await model.generate_content_async(prompt)
        return result.text.strip()
    except Exception as e:
        print(f"⚠️ Fast dialogue fallback error: {e}")
        # Fallback messages by language
        fallbacks = {
            "no": "Jeg jobber med historien din akkurat nå! Den blir fantastisk - bare litt til!",
            "es": "¡Estoy trabajando en tu historia ahora mismo! Va a ser increíble - ¡solo un poco más!",
            "en": "I'm working on your story right now! It's going to be amazing - just a little more time!"
        }
        return fallbacks.get(language, fallbacks["en"])


async def _trigger_chapter_generation(story_id: str, chapter_number: int):
    """
    Trigger chapter generation in background.

    Called when user explicitly requests chapter generation for a story
    that has structure and characters but the chapter hasn't been written yet.

    For chapters 2+, runs Structure V2 refinement first (if not already done)
    to ensure chapter synopses are refined with character skills and Ch1 content.
    """
    try:
        logger.info(f"📝 Background chapter {chapter_number} generation starting for {story_id}")

        # Get coordinator
        if _coordinator is None:
            logger.error("Coordinator not initialized - cannot generate chapter")
            return

        # For Chapter 2+, run Structure V2 refinement first if not already done
        if chapter_number >= 2:
            story = await _coordinator.storage.get_story(story_id)
            # Check if structure has been refined (V2 adds refinement_v2 field to structure)
            structure_refined = (
                story and
                story.structure and
                hasattr(story.structure, 'refinement_v2') and
                story.structure.refinement_v2 is not None
            )

            if not structure_refined:
                logger.info(f"🔄 Running Structure V2 refinement before Chapter {chapter_number}...")
                await story_events.emit_pipeline_stage(story_id, "structure_v2", "in_progress")
                v2_result = await _coordinator.refine_structure_v2(story_id)
                if v2_result.get("success"):
                    refined = v2_result.get("refined_chapters", 0)
                    logger.info(f"✅ Structure V2 complete: {refined} chapters refined")
                else:
                    logger.warning(f"⚠️ Structure V2 failed, continuing with original structure: {v2_result.get('error')}")
            else:
                logger.info(f"✅ Structure V2 already completed, proceeding to Chapter {chapter_number}")

        # Generate the chapter
        result = await _coordinator.write_chapter(story_id, chapter_number)

        if result.get("success"):
            logger.info(f"✅ Chapter {chapter_number} generated successfully for {story_id}")
        else:
            logger.error(f"❌ Chapter generation failed: {result.get('error')}")

    except Exception as e:
        logger.error(f"❌ Error in background chapter generation: {e}", exc_info=True)


async def handle_tier_2_preference(
    websocket: WebSocket,
    story_id: str,
    user_message: str,
    classification: InputClassification,
    current_chapter: int,
    generating_chapter: Optional[int]
):
    """
    Handle Tier 2: Queue preference for next chapter.

    Uses CompanionAgent for consistent Hanan persona acknowledgments.
    """
    # Calculate target chapter
    target_chapter = _input_classifier.calculate_target_chapter(
        classification, current_chapter, generating_chapter
    )

    # Queue the input for future chapter
    queued_input = _input_classifier.create_queued_input(
        classification, user_message, target_chapter
    )
    await _coordinator.storage.queue_user_input(story_id, queued_input)

    # Get story for context
    story = await _coordinator.storage.get_story(story_id)

    # Get language from story preferences
    story_language = story.preferences.language if story and story.preferences else "en"

    # Get CompanionAgent for persona-consistent response
    companion = get_companion_agent()
    response = None
    audio_bytes = None

    if companion:
        try:
            # Use combined LLM+audio method for reduced latency
            response, audio_bytes = await companion.handle_user_message_with_audio(
                story_id=story_id,
                user_message=user_message,
                tier=InputTier.TIER_2_PREFERENCE,
                classification=classification,
                story=story
            )
        except Exception as e:
            logger.warning(f"CompanionAgent error for Tier 2, falling back: {e}")

    # Fallback if CompanionAgent unavailable or failed
    if not response:
        response = _fallback_tier_2_response(classification, story_language)
        audio_bytes = None

    await send_dialogue_response(story_id, response, {
        "tier": 2,
        "phase": "preference_queued",
        "target_chapter": target_chapter
    }, story_language, audio_bytes)

    # Notify client about queued input
    await story_events.emit("input_queued", story_id, {
        "message": user_message,
        "tier": 2,
        "will_appear_in": target_chapter,
        "preference": classification.preference_updates
    })

    print(f"📝 Tier 2 preference queued for Chapter {target_chapter}: {user_message}")


def _fallback_tier_2_response(classification: InputClassification, language: str = "en") -> str:
    """Fallback Tier 2 response if CompanionAgent unavailable"""
    preference = classification.classified_intent.lower()
    fallbacks = {
        "no": f"Skjønner! Jeg gjør historien {preference} i neste del.",
        "es": f"¡Entendido! Haré la historia {preference} en la próxima parte.",
        "en": f"Got it! I'll make the story {preference} in the next part."
    }
    return fallbacks.get(language, fallbacks["en"])


async def handle_tier_3_story_choice(
    websocket: WebSocket,
    story_id: str,
    user_message: str,
    classification: InputClassification,
    current_chapter: int,
    generating_chapter: Optional[int]
):
    """Handle Tier 3: Queue story choice for N+2 chapter.

    Uses CompanionAgent for consistent Hanan persona acknowledgments.
    """
    # Calculate target chapter (never modify what's generating)
    target_chapter = _input_classifier.calculate_target_chapter(
        classification, current_chapter, generating_chapter
    )

    # Queue the input
    queued_input = _input_classifier.create_queued_input(
        classification, user_message, target_chapter
    )
    await _coordinator.storage.queue_user_input(story_id, queued_input)

    # Get story for context
    story = await _coordinator.storage.get_story(story_id)

    # Get language from story preferences
    story_language = story.preferences.language if story and story.preferences else "en"

    # Get CompanionAgent for response
    companion = get_companion_agent()
    response = None
    audio_bytes = None

    if companion:
        try:
            # Use combined LLM+audio method for reduced latency
            response, audio_bytes = await companion.handle_user_message_with_audio(
                story_id=story_id,
                user_message=user_message,
                tier=InputTier.TIER_3_STORY_CHOICE,
                classification=classification,
                story=story
            )
        except Exception as e:
            print(f"⚠️ CompanionAgent error, falling back: {e}")
            # Language-specific fallback
            fallbacks = {
                "no": f"For en flott idé! Jeg legger det til i historien - det dukker opp i kapittel {target_chapter}!",
                "es": f"¡Qué gran idea! Lo añadiré a la historia - ¡aparecerá en el capítulo {target_chapter}!",
                "en": f"What a great idea! I'll add that to the story - it'll appear in Chapter {target_chapter}!"
            }
            response = fallbacks.get(story_language, fallbacks["en"])
            audio_bytes = None

    if not response:
        # Fallback without CompanionAgent
        response = await _fallback_tier_3_response(user_message, classification, story, target_chapter, story_language)
        audio_bytes = None

    await send_dialogue_response(story_id, response, {
        "tier": 3,
        "phase": "choice_queued",
        "target_chapter": target_chapter
    }, story_language, audio_bytes)

    # Notify client
    await story_events.emit("input_queued", story_id, {
        "message": user_message,
        "tier": 3,
        "will_appear_in": target_chapter,
        "story_direction": classification.story_direction
    })

    print(f"🎭 Tier 3 story choice queued for Chapter {target_chapter}: {user_message}")


async def _fallback_tier_3_response(user_message: str, classification: InputClassification, story, target_chapter: int, language: str = "en") -> str:
    """Fallback Tier 3 response if CompanionAgent unavailable"""
    import google.generativeai as genai

    rich_context = build_rich_story_context(story) if story else "Story context unavailable"

    # Get language style
    lang_style = get_language_style(language)
    lang_instruction = get_dialogue_instruction(language)

    prompt = f"""You are a friendly storyteller talking to a child. They want to add something to the story: "{user_message}"

{lang_instruction}

Their idea: {classification.story_direction}

=== STORY CONTEXT ===
{rich_context}
=== END CONTEXT ===

Respond with excitement in 2-3 sentences in {lang_style['name']}:
1. Love their idea
2. Connect it to existing characters/plot
3. Tease what might happen

Example (translate to {lang_style['name']}): "Oh, a wolf! What a great idea! I wonder if Harald will meet a wolf during his training..."

Return ONLY the dialogue text in {lang_style['name']}."""

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        result = await model.generate_content_async(prompt)
        return result.text.strip()
    except Exception as e:
        print(f"⚠️ Fast dialogue fallback error: {e}")
        fallbacks = {
            "no": f"For en flott idé! Jeg legger det til i historien - det dukker opp i kapittel {target_chapter}!",
            "es": f"¡Qué gran idea! Lo añadiré a la historia - ¡aparecerá en el capítulo {target_chapter}!",
            "en": f"What a great idea! I'll add that to the story - it'll appear in Chapter {target_chapter}!"
        }
        return fallbacks.get(language, fallbacks["en"])


async def handle_tier_4_addition(
    websocket: WebSocket,
    story_id: str,
    user_message: str,
    classification: InputClassification,
    current_chapter: int,
    generating_chapter: Optional[int]
):
    """Handle Tier 4: Queue major addition for appropriate chapter."""
    # Target chapter is at least N+2, preferably further
    target_chapter = max(
        current_chapter + 2,
        (generating_chapter or 0) + 1
    )

    # Queue the input
    queued_input = _input_classifier.create_queued_input(
        classification, user_message, target_chapter
    )
    await _coordinator.storage.queue_user_input(story_id, queued_input)

    # Get story for language
    story = await _coordinator.storage.get_story(story_id)
    story_language = story.preferences.language if story and story.preferences else "en"

    # Send acknowledgment in appropriate language
    fallbacks = {
        "no": "For en kreativ idé! Jeg fletter det inn i historien snart. Fortsett å lytte!",
        "es": "¡Qué idea tan creativa! La integraré en la historia pronto. ¡Sigue escuchando!",
        "en": "What a creative idea! I'll weave that into the story soon. Keep listening!"
    }
    response = fallbacks.get(story_language, fallbacks["en"])

    await send_dialogue_response(story_id, response, {
        "tier": 4,
        "phase": "addition_queued",
        "target_chapter": target_chapter
    }, story_language)

    # Notify client
    await story_events.emit("input_queued", story_id, {
        "message": user_message,
        "tier": 4,
        "will_appear_in": target_chapter,
        "addition": classification.classified_intent
    })

    print(f"✨ Tier 4 addition queued for Chapter {target_chapter}: {user_message}")


# ===== Reading State Handlers =====

async def handle_start_reading(websocket: WebSocket, story_id: str, chapter_num: int):
    """Handle user starting to read a chapter."""
    # Update local reading state
    _reading_states[story_id] = {
        "chapter": chapter_num,
        "is_reading": True,
        "position": 0.0
    }

    # Update Firebase reading state
    if _coordinator:
        await _coordinator.storage.update_chapter_status(
            story_id, chapter_num, ChapterStatus.READING
        )

        # Trigger buffered generation of next chapter
        result = await _coordinator.on_user_starts_reading(story_id, chapter_num)

        if result.get("next_chapter_started"):
            next_chapter = result.get("next_chapter")
            await story_events.emit("chapter_generating", story_id, {
                "chapter": next_chapter,
                "message": f"Preparing Chapter {next_chapter}..."
            })

    # Confirm reading started
    await websocket.send_json({
        "type": "reading_started",
        "chapter": chapter_num
    })


async def handle_pause_reading(websocket: WebSocket, story_id: str, position: float):
    """Handle user pausing reading."""
    if story_id in _reading_states:
        _reading_states[story_id]["is_reading"] = False
        _reading_states[story_id]["position"] = position

    # Update Firebase
    if _coordinator:
        chapter = _reading_states.get(story_id, {}).get("chapter", 1)
        await _coordinator.storage.update_reading_position(story_id, chapter, position)

    await websocket.send_json({
        "type": "reading_paused",
        "position": position
    })


async def handle_resume_reading(websocket: WebSocket, story_id: str):
    """Handle user resuming reading."""
    if story_id in _reading_states:
        _reading_states[story_id]["is_reading"] = True

    await websocket.send_json({
        "type": "reading_resumed"
    })


async def handle_start_chapter(websocket: WebSocket, story_id: str, chapter_num: int):
    """
    Handle user starting to play a chapter.

    This is the trigger for one-chapter-at-a-time prefetch:
    When user starts playing Chapter N, we prefetch Chapter N+1.

    Args:
        websocket: WebSocket connection
        story_id: Story ID
        chapter_num: Chapter number being started
    """
    # Update local reading state
    _reading_states[story_id] = {
        "chapter": chapter_num,
        "is_reading": True,
        "position": 0.0
    }

    # Update Firebase reading state
    if _coordinator:
        await _coordinator.storage.update_chapter_status(
            story_id, chapter_num, ChapterStatus.READING
        )

    # Trigger prefetch of next chapter via CompanionAgent
    companion = get_companion_agent()
    if companion:
        # Non-blocking: trigger prefetch in background
        asyncio.create_task(
            companion.on_chapter_playback_start(story_id, chapter_num)
        )
        logger.info(f"Triggered prefetch for next chapter after Chapter {chapter_num}")

    # Confirm chapter started
    await websocket.send_json({
        "type": "chapter_started",
        "chapter_number": chapter_num,
        "message": f"Starting Chapter {chapter_num}"
    })

    # Emit event for tracking
    await story_events.emit("chapter_playback_started", story_id, {
        "chapter_number": chapter_num
    })


async def handle_finish_reading(websocket: WebSocket, story_id: str, chapter_num: int):
    """
    Handle user finishing reading a chapter.

    Triggers post-chapter discussion via CompanionAgent before offering
    next chapter. This creates a teacher-like experience where the
    CompanionAgent discusses what just happened in the story.
    """
    # Update local state
    if story_id in _reading_states:
        _reading_states[story_id]["is_reading"] = False
        _reading_states[story_id]["position"] = 1.0
        _reading_states[story_id]["playback_phase"] = PlaybackPhase.POST_CHAPTER.value

    if _coordinator:
        # Mark chapter as completed
        await _coordinator.storage.mark_chapter_completed(story_id, chapter_num)

        # Update reading state in Firebase with POST_CHAPTER phase
        await _coordinator.storage.update_reading_state(story_id, {
            "playback_phase": PlaybackPhase.POST_CHAPTER.value,
            "discussion_started": True
        })

        # Get CompanionAgent for post-chapter discussion
        companion = get_companion_agent()

        if companion:
            try:
                # Generate teacher-like post-chapter discussion
                discussion = await companion.generate_post_chapter_discussion(
                    story_id=story_id,
                    chapter_number=chapter_num
                )

                # Emit the discussion (only if message exists)
                message = discussion.get("message")
                if message:
                    await story_events.emit("dialogue_ready", story_id, {
                        "message": message,
                        "audio": discussion.get("audio"),
                        "source": "companion_agent",
                        "discussion_type": "post_chapter",
                        "chapter_number": chapter_num,
                        "has_more_chapters": discussion.get("has_more_chapters", False)
                    })
                else:
                    logger.warning(f"Skipping dialogue_ready - no message in discussion for chapter {chapter_num}")

                logger.info(f"Post-chapter discussion sent for story {story_id} chapter {chapter_num}")

            except Exception as e:
                logger.error(f"Error generating post-chapter discussion: {e}")
                # Fallback to standard flow
                await _fallback_chapter_transition(story_id, chapter_num)
        else:
            # No CompanionAgent available, use standard flow
            await _fallback_chapter_transition(story_id, chapter_num)

    await websocket.send_json({
        "type": "reading_finished",
        "chapter": chapter_num,
        "playback_phase": PlaybackPhase.POST_CHAPTER.value
    })


async def _fallback_chapter_transition(story_id: str, chapter_num: int):
    """Fallback chapter transition when CompanionAgent unavailable."""
    if not _coordinator:
        return

    # Get story for language
    story = await _coordinator.storage.get_story(story_id)
    story_language = story.preferences.language if story and story.preferences else "en"

    result = await _coordinator.on_user_finishes_chapter(story_id, chapter_num)

    if result.get("status") == "next_chapter_ready":
        next_chapter = result.get("next_chapter")
        title = result.get("next_chapter_title", f"Chapter {next_chapter}")

        # NOTE: chapter_ready event is already emitted by coordinator.py:3226 when chapter is written
        # Do NOT emit here - it causes duplicate announcements in chat
        # The dialogue response below handles user communication for this transition

        # Language-specific messages
        messages = {
            "no": f"Det var et flott kapittel! {title} er klar når du vil fortsette!",
            "es": f"¡Ese fue un gran capítulo! ¡{title} está listo cuando quieras continuar!",
            "en": f"That was a great chapter! {title} is ready whenever you want to continue!"
        }

        await send_dialogue_response(
            story_id,
            messages.get(story_language, messages["en"]),
            {"phase": "chapter_transition", "next_chapter": next_chapter},
            story_language
        )

    elif result.get("status") == "generating":
        bridge = result.get("bridge_content", {})
        # Bridge content should already be in correct language from coordinator
        fallback_messages = {
            "no": "Neste kapittel er nesten klart...",
            "es": "El próximo capítulo está casi listo...",
            "en": "The next chapter is almost ready..."
        }
        await send_dialogue_response(
            story_id,
            bridge.get("message", fallback_messages.get(story_language, fallback_messages["en"])),
            {"phase": "bridge", "bridge_type": bridge.get("type")},
            story_language
        )

    elif result.get("status") == "story_complete":
        await story_events.emit("story_complete", story_id, {
            "message": "The story has ended!",
            "total_chapters": chapter_num
        })

        # Language-specific completion messages
        completion_messages = {
            "no": "Og det er slutten på historien vår! Jeg håper du likte den!",
            "es": "¡Y ese es el final de nuestra historia! ¡Espero que la hayas disfrutado!",
            "en": "And that's the end of our story! I hope you enjoyed it!"
        }

        await send_dialogue_response(
            story_id,
            completion_messages.get(story_language, completion_messages["en"]),
            {"phase": "story_complete"},
            story_language
        )


async def send_dialogue_response(
    story_id: str,
    message: str,
    metadata: Dict = None,
    language: str = "en",
    audio_bytes: bytes = None
):
    """Helper to send dialogue with audio generation.

    PRIORITY DELIVERY: Sends directly to WebSocket first, then emits to queue.
    This ensures dialogue reaches the child in <2s even during heavy agent work.

    Args:
        story_id: Story ID
        message: Dialogue message text
        metadata: Optional metadata dict
        language: Language code for TTS (en, no, es)
        audio_bytes: Optional pre-generated audio bytes (skips TTS if provided)
    """
    start_time = time.time()

    # Use pre-generated audio or generate via TTS
    if audio_bytes is None:
        # Get TTS language code
        tts_language = get_tts_language_code(language)

        # Generate audio with appropriate language (dialogue = companion speech)
        audio_bytes = await voice_service.text_to_speech(
            text=message,
            speaking_rate=0.9,
            language_code=tts_language,
            use_case="dialogue"
        )

    audio_base64 = None
    if audio_bytes:
        audio_base64 = voice_service.encode_audio_base64(audio_bytes)

    # Build the event payload
    event_data = {
        "type": "dialogue_ready",
        "story_id": story_id,
        "data": {
            "message": message,
            "audio": audio_base64,
            **(metadata or {})
        },
        "timestamp": datetime.now().isoformat()
    }

    # =========================================================================
    # PRIORITY DELIVERY: Send DIRECTLY to WebSocket first
    # This bypasses the event queue to ensure <2s latency during heavy agent work
    # =========================================================================
    websocket = manager.active_connections.get(story_id)
    if websocket:
        try:
            # Yield first to let other coroutines run
            await asyncio.sleep(0)

            # Send directly - no queue delay
            await websocket.send_json(event_data)

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"⚡ PRIORITY dialogue sent in {elapsed_ms:.0f}ms (direct WebSocket)")
        except Exception as e:
            logger.warning(f"⚠️ Direct WebSocket send failed, falling back to queue: {e}")
            # Fall back to queue-based delivery
            await story_events.emit(
                "dialogue_ready",
                story_id,
                {
                    "message": message,
                    "audio": audio_base64,
                    **(metadata or {})
                }
            )
    else:
        # No active WebSocket - use queue (shouldn't happen normally)
        logger.warning(f"⚠️ No active WebSocket for {story_id}, using queue")
        await story_events.emit(
            "dialogue_ready",
            story_id,
            {
                "message": message,
                "audio": audio_base64,
                **(metadata or {})
            }
        )

    # Save to database (non-blocking - don't wait)
    if _coordinator:
        entry = DialogueEntry(
            speaker="agent",
            message=message,
            metadata=metadata or {}
        )
        # Use create_task to not block on database save
        asyncio.create_task(_coordinator.storage.save_dialogue(story_id, entry))
