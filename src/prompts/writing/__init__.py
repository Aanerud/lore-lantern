"""
Writing Prompts Package

This package contains prompts for the chapter writing workflow:
- write_chapter: Initial chapter drafting (Nnedi Okofor persona)
- revise_chapter: Round Table revision based on feedback
- polish_chapter: Final polish pass with line edits

Each prompt is a function that accepts context and returns a formatted prompt string.
"""

from .write_chapter import get_write_chapter_prompt, get_write_chapter_json_schema
from .revise_chapter import get_revise_chapter_prompt
from .polish_chapter import get_polish_chapter_prompt

__all__ = [
    "get_write_chapter_prompt",
    "get_write_chapter_json_schema",
    "get_revise_chapter_prompt",
    "get_polish_chapter_prompt",
]
