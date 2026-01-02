"""
Dialogue Agent - The Enthusiastic Teacher (Hanan persona)

This agent is the user-facing personality that keeps children engaged,
answers questions, and guides the storytelling experience.

Model configuration is loaded from src/config/models.yaml via LLMRouter.
"""

from crewai import Agent, LLM
from pathlib import Path
from typing import Optional


def create_dialogue_agent(model_override: Optional[str] = None) -> Agent:
    """
    Create the DialogueAgent - an enthusiastic teacher personality.

    Args:
        model_override: Optional model to use instead of config.
                        If None, uses model from models.yaml

    Returns:
        CrewAI Agent configured as enthusiastic teacher
    """
    from src.services.llm_router import get_llm_router

    router = get_llm_router()
    llm_kwargs = router.get_llm_kwargs("dialogue", model_override=model_override)

    # Load system prompt from src/prompts/
    prompt_path = Path(__file__).parent.parent / "prompts" / "dialogue_agent.txt"
    with open(prompt_path, 'r') as f:
        system_prompt = f.read()

    return Agent(
        role='Enthusiastic Teacher',
        goal='Keep child engaged, excited about learning, and create memorable storytelling experiences',
        backstory=system_prompt,
        llm=LLM(**llm_kwargs),
        verbose=True,
        allow_delegation=False,  # Dialogue agent responds directly, doesn't delegate
        max_iter=3,  # Quick responses
    )
