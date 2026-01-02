"""
Model Tracker - Tracks models used by agents with intelligent fallback

For pre-configured models (claude-sonnet-4-5, gemini-3-flash-preview, etc.),
we use the configured model from LLM Router since we already know what's being used.

For Azure AI Foundry's model-router, the actual model selected
(e.g., "gpt-oss-120b", "DeepSeek-V3.1", "gpt-5-chat-2025-08-07")
is captured via LiteLLM callback from the API response.

This hybrid approach ensures:
1. Pre-set models always show correctly (no callback dependency)
2. Model-router captures the actual dynamic selection
3. Thread-safe tracking for parallel agent execution

Usage:
    from src.services.model_tracker import init_model_tracker, ModelTracker

    # At app startup
    init_model_tracker()

    # Before running an agent - pass the configured model
    tracker = ModelTracker.get_instance()
    tracker.set_current_agent("CharacterAgent", configured_model="gemini/gemini-3-flash-preview")

    # Run agent via CrewAI/LiteLLM
    result = crew.kickoff()

    # After - get model used (configured or callback-captured for model-router)
    actual_model = tracker.get_last_model("CharacterAgent")
    # Returns "gemini/gemini-3-flash-preview" or "gpt-5-chat-2025-08-07" for model-router
"""

import logging
import threading
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Thread-local storage for current agent context
# Each ThreadPoolExecutor worker thread gets its own storage
# This ensures parallel reviewers don't overwrite each other's context
_thread_local = threading.local()


class ModelTracker:
    """
    Thread-safe singleton that tracks models used by each agent.

    Supports two model sources:
    1. Configured model - from LLM Router config (always available)
    2. Callback model - captured from LiteLLM response (for model-router)

    For pre-set models, the configured model is used.
    For model-router, the callback captures the actual selected model.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._configured_models: Dict[str, str] = {}  # agent_name -> configured model
        self._callback_models: Dict[str, str] = {}    # agent_name -> callback-captured model
        self._agent_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'ModelTracker':
        """Get the singleton instance."""
        return cls()

    def set_current_agent(self, agent_name: str, configured_model: Optional[str] = None) -> None:
        """
        Set the current agent name and optionally its configured model.

        Call this BEFORE starting a CrewAI/LiteLLM task so the callback
        knows which agent to associate the model with.

        Uses threading.local() for thread-safe tracking - each thread
        maintains its own current_agent value, preventing race conditions
        when multiple agents run in parallel (e.g., Round Table reviewers
        via ThreadPoolExecutor).

        Args:
            agent_name: Name like "CharacterAgent", "Reviewer_Guillermo", etc.
            configured_model: The model configured in LLM Router for this agent.
                              Used as fallback when callback doesn't capture model.
        """
        _thread_local.current_agent = agent_name
        _thread_local.configured_model = configured_model

        # Store configured model (this is thread-safe via dict)
        if configured_model:
            with self._agent_lock:
                self._configured_models[agent_name] = configured_model

        logger.debug(f"ModelTracker: Agent '{agent_name}' set with model '{configured_model}' (thread {threading.current_thread().name})")

    def clear_current_agent(self) -> None:
        """Clear the current agent (call after task completes)."""
        _thread_local.current_agent = None
        _thread_local.configured_model = None

    def record_model(self, model: str) -> None:
        """
        Record the callback-captured model for the current agent.

        Called by the LiteLLM callback after each successful completion.
        This is primarily useful for model-router where the actual model
        (e.g., "gpt-5-chat-2025-08-07") differs from the configured model.

        For pre-set models, this may capture the same model or not be called
        at all (depends on provider response format).

        Args:
            model: The actual model name from response.model
        """
        agent_name = getattr(_thread_local, 'current_agent', None)
        if agent_name and model:
            with self._agent_lock:
                self._callback_models[agent_name] = model
            logger.info(f"ModelTracker: {agent_name} callback captured model '{model}'")
        else:
            logger.debug(f"ModelTracker: Model '{model}' but no current agent (thread {threading.current_thread().name})")

    def get_last_model(self, agent_name: str) -> Optional[str]:
        """
        Get the model used by a specific agent.

        Priority:
        1. Callback-captured model (for model-router dynamic selection)
        2. Configured model (from LLM Router config)
        3. None if neither available

        Args:
            agent_name: The agent name to look up

        Returns:
            Model name like "gpt-5-chat-2025-08-07" or "gemini/gemini-3-flash-preview"
        """
        # For model-router, callback captures the actual model
        callback_model = self._callback_models.get(agent_name)
        if callback_model:
            return callback_model

        # For pre-set models, use the configured model
        return self._configured_models.get(agent_name)

    def get_all_models(self) -> Dict[str, str]:
        """
        Get all agent -> model mappings.

        Merges configured and callback-captured models, with callback taking priority.
        """
        result = dict(self._configured_models)
        result.update(self._callback_models)  # Callback overrides configured
        return result

    def is_model_router(self, agent_name: str) -> bool:
        """Check if an agent is using model-router (dynamic model selection)."""
        configured = self._configured_models.get(agent_name, "")
        return "model-router" in configured.lower()


def model_tracker_callback(
    kwargs: Dict[str, Any],
    completion_response: Any,
    start_time: Any,
    end_time: Any,
) -> None:
    """
    LiteLLM success callback - captures response.model after each completion.

    This is called by LiteLLM after every successful API call. We extract
    the actual model name from the response and record it.

    Args:
        kwargs: The original kwargs passed to litellm.completion()
        completion_response: The response object from the API
        start_time: When the request started
        end_time: When the response was received
    """
    try:
        tracker = ModelTracker.get_instance()
        if hasattr(completion_response, 'model') and completion_response.model:
            tracker.record_model(completion_response.model)
    except Exception as e:
        logger.warning(f"ModelTracker callback error: {e}")


def init_model_tracker() -> None:
    """
    Initialize the model tracker and register LiteLLM callback.

    Call this once at application startup, after LiteLLM environment
    variables are configured.
    """
    try:
        import litellm

        # Get existing callbacks (if any) and add ours
        existing = getattr(litellm, 'success_callback', None) or []
        if not isinstance(existing, list):
            existing = [existing] if existing else []

        # Add our callback if not already present
        if model_tracker_callback not in existing:
            existing.append(model_tracker_callback)
            litellm.success_callback = existing
            logger.info("ModelTracker: LiteLLM callback registered")
        else:
            logger.debug("ModelTracker: Callback already registered")

    except ImportError:
        logger.warning("ModelTracker: LiteLLM not installed, callback not registered")
    except Exception as e:
        logger.error(f"ModelTracker: Failed to register callback: {e}")
