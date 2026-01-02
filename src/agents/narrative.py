"""
Narrative Agent - The Story Writer (Africanfuturist persona)

Writes engaging, age-appropriate narrative content with educational elements
woven naturally into the story.

Model configuration is loaded from src/config/models.yaml via LLMRouter.
"""

from pathlib import Path
from crewai import Agent, LLM
from typing import Optional


def _load_narrative_prompt() -> str:
    """Load the narrative agent prompt from external file."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "narrative_agent.txt"
    if prompt_path.exists():
        return prompt_path.read_text()
    else:
        raise FileNotFoundError(f"Narrative agent prompt not found at {prompt_path}")


def create_narrative_agent(model_override: Optional[str] = None) -> Agent:
    """
    Create the NarrativeAgent - the actual story writer.

    Args:
        model_override: Optional model to use instead of config.
                        If None, uses model from models.yaml

    Returns:
        CrewAI Agent configured for narrative writing
    """
    from src.services.llm_router import get_llm_router

    router = get_llm_router()
    llm_kwargs = router.get_llm_kwargs("narrative", model_override=model_override)

    # Load the narrative prompt from external file
    backstory = _load_narrative_prompt()

    return Agent(
        role='Africanfuturist Story Writer',
        goal='Write beautiful, engaging prose that captivates young readers while teaching them new concepts, expanding their worldview, and centering diverse perspectives that challenge Western-centric narratives',
        backstory=backstory,
        llm=LLM(**llm_kwargs),
        verbose=True,
        allow_delegation=False,
        max_iter=5
    )
