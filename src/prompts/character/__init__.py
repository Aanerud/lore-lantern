"""
Character Prompts Package

This package contains prompts for character creation and validation:
- create_character: D&D-style character creation with skills and arc
- validate_character: Bill Nye historical accuracy and duplicate checking

Each prompt is a function that accepts context and returns a formatted prompt string.
"""

from .create_character import get_create_character_prompt
from .validate_character import get_validate_character_prompt

__all__ = [
    "get_create_character_prompt",
    "get_validate_character_prompt",
]
