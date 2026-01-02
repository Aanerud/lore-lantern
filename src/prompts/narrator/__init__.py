"""
Narrator Prompts Package

This package contains prompts for narrator-related tasks:
- narrative_method_debate: Guillermo vs Stephen POV discussion
- discussion: Writers' room discussion facilitation
- commentary: Event-specific narrator commentary

Each prompt is a function that accepts context and returns a formatted prompt string.
"""

from .narrative_method_debate import get_narrative_method_debate_prompt
from .discussion import get_discussion_prompt
from .commentary import (
    get_structure_ready_commentary_prompt,
    get_character_ready_commentary_prompt,
    get_character_ready_fallback_prompt,
    get_chapter_ready_commentary_prompt,
    get_chapter_ready_fallback_prompt,
    get_fallback_commentary_prompt
)

__all__ = [
    "get_narrative_method_debate_prompt",
    "get_discussion_prompt",
    "get_structure_ready_commentary_prompt",
    "get_character_ready_commentary_prompt",
    "get_character_ready_fallback_prompt",
    "get_chapter_ready_commentary_prompt",
    "get_chapter_ready_fallback_prompt",
    "get_fallback_commentary_prompt",
]
