"""
Round Table Review Prompts

This package contains prompts for the 6 Round Table reviewers:
- Guillermo del Toro (Structure)
- Bill Nye (Facts & Education)
- Clarissa Pinkola Estés (Character Psychology)
- Benjamin Dreyer (Prose Quality)
- Stephen King (Tension & Hooks)
- Continuity Editor (Plot Threads)

Each prompt is a function that accepts context and returns a formatted prompt string.
"""

from .review_guillermo_structure import get_guillermo_structure_prompt
from .review_bill_facts import get_bill_facts_prompt
from .review_clarissa_characters import get_clarissa_characters_prompt
from .review_benjamin_prose import get_benjamin_prose_prompt
from .review_stephen_tension import get_stephen_tension_prompt
from .review_continuity_threads import get_continuity_threads_prompt

__all__ = [
    "get_guillermo_structure_prompt",
    "get_bill_facts_prompt",
    "get_clarissa_characters_prompt",
    "get_benjamin_prose_prompt",
    "get_stephen_tension_prompt",
    "get_continuity_threads_prompt",
]
