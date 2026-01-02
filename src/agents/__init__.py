"""Agents package for Kids Storyteller V2"""

from .dialogue import create_dialogue_agent
from .structure import create_structure_agent
from .character import create_character_agent
from .narrative import create_narrative_agent
from .factcheck import create_factcheck_agent
from .voice_director import create_voice_director_agent
from .tension import create_tension_agent


def get_llm_kwargs(
    model: str,
    temperature: float,
    top_p: float = None,
    **kwargs
) -> dict:
    """
    Build LLM kwargs with Claude-safe parameter handling.

    Claude models don't allow both temperature AND top_p together.
    This helper conditionally includes top_p only for non-Claude models.

    Args:
        model: Model name (e.g., "claude-sonnet-4-5-20250929", "azure/model-router")
        temperature: Temperature value (always included)
        top_p: Top-p value (only included for non-Claude models)
        **kwargs: Additional LLM parameters (max_tokens, timeout, etc.)

    Returns:
        Dict of LLM kwargs ready for CrewAI LLM constructor
    """
    # Check if using Claude model (any variant via Anthropic or Azure)
    is_claude = "claude" in model.lower() or "anthropic" in model.lower()

    result = {
        "model": model,
        "temperature": temperature,
        "drop_params": True,  # Auto-drop unsupported params
        **kwargs
    }

    # Only add top_p for non-Claude models (OpenAI, Gemini, Azure model-router)
    # Claude models reject requests that have both temperature AND top_p
    if top_p is not None and not is_claude:
        result["top_p"] = top_p

    return result


__all__ = [
    "create_dialogue_agent",
    "create_structure_agent",
    "create_character_agent",
    "create_narrative_agent",
    "create_factcheck_agent",
    "create_voice_director_agent",
    "create_tension_agent",
    "get_llm_kwargs",
]
