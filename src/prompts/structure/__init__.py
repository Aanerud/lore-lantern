"""
Structure Prompts Package

This package contains prompts for story structure generation:
- generate_story_structure: Initial story outline with character planning
- refine_structure_v2: Synopsis refinement after Chapter 1

Each prompt is a function that accepts context and returns a formatted prompt string.
"""

from .generate_structure import get_generate_structure_prompt
from .refine_structure import get_refine_structure_prompt, get_refine_structure_system_prompt

__all__ = [
    "get_generate_structure_prompt",
    "get_refine_structure_prompt",
    "get_refine_structure_system_prompt",
]
