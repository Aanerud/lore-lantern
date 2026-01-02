"""
Event System for Real-Time Story Updates

Manages event emissions and WebSocket broadcasting for story progress.
Includes observatory events for debug panel visualization.
"""

from typing import Dict, Callable, Any, List, Optional
from asyncio import Queue
import asyncio
from datetime import datetime
import time


# ==================== Event Type Constants ====================
# Standard story events
EVENT_DIALOGUE_READY = "dialogue_ready"
EVENT_DIALOGUE_CHUNK = "dialogue_chunk"  # Streaming text chunk
EVENT_DIALOGUE_TEXT_COMPLETE = "dialogue_text_complete"  # Streaming text finished
EVENT_DIALOGUE_AUDIO_READY = "dialogue_audio_ready"  # Audio arrives after text
EVENT_STRUCTURE_READY = "structure_ready"
EVENT_CHARACTER_READY = "character_ready"
EVENT_CHAPTER_READY = "chapter_ready"
EVENT_CHAPTER_GENERATING = "chapter_generating"
EVENT_SSML_READY = "ssml_ready"
EVENT_STORY_COMPLETE = "story_complete"

# Observatory events (for debug panel)
EVENT_AGENT_STARTED = "agent_started"
EVENT_AGENT_PROGRESS = "agent_progress"
EVENT_AGENT_COMPLETED = "agent_completed"
EVENT_MODEL_SELECTED = "model_selected"
EVENT_MODEL_RESPONSE = "model_response"
EVENT_ROUND_TABLE_STARTED = "round_table_started"
EVENT_REVIEWER_WORKING = "reviewer_working"
EVENT_REVIEWER_VERDICT = "reviewer_verdict"
EVENT_ROUND_TABLE_DECISION = "round_table_decision"
EVENT_PIPELINE_STAGE = "pipeline_stage"


class StoryEvent:
    """Represents a story event"""
    def __init__(self, event_type: str, story_id: str, data: Dict[str, Any]):
        self.event_type = event_type
        self.story_id = story_id
        self.data = data
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.event_type,
            "story_id": self.story_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }


class EventEmitter:
    """
    Event emitter for story progress events.

    Emits events when story components are ready:
    - dialogue_ready: DialogueAgent has new message
    - structure_ready: Story structure completed
    - character_ready: Character created/updated
    - chapter_ready: Chapter written and fact-checked
    - story_complete: All chapters finished
    """

    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}
        self._story_queues: Dict[str, Queue] = {}  # story_id -> event queue
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Store reference to main event loop for thread-safe emission."""
        self._main_loop = loop

    def on(self, event_type: str, callback: Callable):
        """Register event listener"""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)

    def off(self, event_type: str, callback: Callable):
        """Remove event listener"""
        if event_type in self._listeners:
            self._listeners[event_type].remove(callback)

    async def emit(self, event_type: str, story_id: str, data: Dict[str, Any]):
        """Emit event to all registered listeners"""
        event = StoryEvent(event_type, story_id, data)

        # Call registered listeners
        if event_type in self._listeners:
            for callback in self._listeners[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    print(f"Error in event listener for {event_type}: {e}")

        # Add to story-specific queue for WebSocket streaming
        if story_id in self._story_queues:
            await self._story_queues[story_id].put(event)

    def emit_from_thread(self, event_type: str, story_id: str, data: Dict[str, Any]):
        """
        Fire-and-forget event emission from worker threads.

        Use this when emitting events from synchronous code running in a thread pool
        (e.g., CrewAI agent work). This does NOT block the calling thread.

        Args:
            event_type: Type of event to emit
            story_id: Story ID for the event
            data: Event data dictionary
        """
        try:
            # Get the main event loop
            loop = self._main_loop
            if loop is None:
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # No event loop available - log and return
                    print(f"[EventEmitter] No event loop for emit_from_thread: {event_type}")
                    return

            # Schedule the coroutine on the main loop without waiting
            asyncio.run_coroutine_threadsafe(
                self.emit(event_type, story_id, data),
                loop
            )
            # Don't call future.result() - that would block!

        except Exception as e:
            # Non-blocking means we can't raise - just log
            print(f"[EventEmitter] emit_from_thread failed ({event_type}): {e}")

    def create_story_queue(self, story_id: str) -> Queue:
        """Create event queue for a specific story (for WebSocket connection)"""
        queue = Queue()
        self._story_queues[story_id] = queue
        return queue

    def remove_story_queue(self, story_id: str):
        """Remove story queue when WebSocket disconnects"""
        if story_id in self._story_queues:
            del self._story_queues[story_id]

    # ==================== Observatory Event Helpers ====================
    # These helper methods make it easy to emit structured observatory events

    async def emit_agent_started(
        self,
        story_id: str,
        agent_name: str,
        task: str,
        metadata: Optional[Dict] = None,
        model: Optional[str] = None
    ):
        """Emit when an agent starts working on a task."""
        data = {
            "agent": agent_name,
            "task": task,
            "started_at": datetime.now().isoformat(),
            **(metadata or {})
        }
        if model:
            data["model"] = model
        await self.emit(EVENT_AGENT_STARTED, story_id, data)

    async def emit_agent_progress(
        self,
        story_id: str,
        agent_name: str,
        progress: int,
        step: str
    ):
        """Emit agent progress update (0-100%)."""
        await self.emit(EVENT_AGENT_PROGRESS, story_id, {
            "agent": agent_name,
            "progress": progress,
            "step": step
        })

    async def emit_agent_completed(
        self,
        story_id: str,
        agent_name: str,
        duration_ms: int,
        success: bool,
        result_summary: Optional[str] = None,
        model: Optional[str] = None
    ):
        """Emit when an agent completes its task."""
        await self.emit(EVENT_AGENT_COMPLETED, story_id, {
            "agent": agent_name,
            "duration_ms": duration_ms,
            "success": success,
            "result_summary": result_summary,
            "model": model  # Actual model used (e.g., "gpt-oss-120b")
        })

    async def emit_model_selected(
        self,
        story_id: str,
        agent_name: str,
        model: str,
        mode: str,
        reason: Optional[str] = None
    ):
        """Emit when model router selects a model for an agent."""
        await self.emit(EVENT_MODEL_SELECTED, story_id, {
            "agent": agent_name,
            "model": model,
            "mode": mode,
            "reason": reason
        })

    async def emit_model_response(
        self,
        story_id: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: int
    ):
        """Emit after receiving model response with usage stats."""
        await self.emit(EVENT_MODEL_RESPONSE, story_id, {
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms
        })

    async def emit_round_table_started(
        self,
        story_id: str,
        chapter_number: int,
        reviewers: List[str]
    ):
        """Emit when Round Table review begins for a chapter."""
        await self.emit(EVENT_ROUND_TABLE_STARTED, story_id, {
            "chapter": chapter_number,
            "reviewers": reviewers,
            "started_at": datetime.now().isoformat()
        })

    async def emit_reviewer_working(
        self,
        story_id: str,
        chapter_number: int,
        reviewer: str
    ):
        """Emit when a reviewer's LLM call starts (before crew.kickoff)."""
        await self.emit(EVENT_REVIEWER_WORKING, story_id, {
            "chapter": chapter_number,
            "reviewer": reviewer,
            "started_at": datetime.now().isoformat()
        })

    async def emit_reviewer_verdict(
        self,
        story_id: str,
        chapter_number: int,
        reviewer: str,
        verdict: str,
        notes: Optional[str] = None,
        model: Optional[str] = None,
        domain: Optional[str] = None,
        duration_ms: Optional[int] = None
    ):
        """Emit when a Round Table reviewer gives their verdict."""
        await self.emit(EVENT_REVIEWER_VERDICT, story_id, {
            "chapter": chapter_number,
            "reviewer": reviewer,
            "verdict": verdict,  # "approve", "concern", "block"
            "notes": notes,
            "model": model,  # Actual model used (e.g., "gpt-oss-120b")
            "domain": domain,  # Review domain (e.g., "structure", "facts")
            "duration_ms": duration_ms  # Time taken for LLM call
        })

    async def emit_round_table_decision(
        self,
        story_id: str,
        chapter_number: int,
        decision: str,
        revision_round: int,
        summary: Optional[str] = None
    ):
        """Emit final Round Table decision for a chapter."""
        await self.emit(EVENT_ROUND_TABLE_DECISION, story_id, {
            "chapter": chapter_number,
            "decision": decision,  # "approved", "revision_needed"
            "revision_round": revision_round,
            "summary": summary
        })

    async def emit_pipeline_stage(
        self,
        story_id: str,
        stage: str,
        status: str,
        progress: Optional[str] = None,
        details: Optional[Dict] = None
    ):
        """Emit pipeline stage update."""
        await self.emit(EVENT_PIPELINE_STAGE, story_id, {
            "stage": stage,  # "structure", "characters", "chapter_N", "round_table", "polish"
            "status": status,  # "pending", "in_progress", "completed", "error"
            "progress": progress,  # e.g., "2/5"
            **(details or {})
        })


# Global event emitter instance
story_events = EventEmitter()
