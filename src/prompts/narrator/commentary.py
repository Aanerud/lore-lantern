"""
Narrator Commentary Prompts

Event-specific narrator commentary for real-time updates:
- structure_ready: Story outline is complete
- character_ready: A new character was created
- chapter_ready: A chapter was written

Each prompt generates brief, warm narrator commentary in the story's language.
"""

from typing import Dict, Any, Optional


def get_structure_ready_commentary_prompt(
    title: str,
    theme: str,
    chapter_count: int,
    chapter_details: str,
    educational_goals: str,
    lang_name: str,
    style_instruction: str
) -> str:
    """
    Generate commentary prompt when story structure is ready.

    Args:
        title: Story title
        theme: Story theme
        chapter_count: Number of chapters
        chapter_details: Formatted chapter preview
        educational_goals: Formatted educational goals
        lang_name: Language name (e.g., "Norwegian")
        style_instruction: Language-specific style instruction

    Returns:
        Formatted prompt string
    """
    return f"""Great news! The story structure is ready!

Story: "{title}"
Theme: {theme}
Total chapters: {chapter_count}{educational_goals}{chapter_details}

=== LANGUAGE REQUIREMENT ===
IMPORTANT: You MUST respond in {lang_name}.
{style_instruction}

Generate a brief (2-3 sentences) warm, natural comment about the story structure IN {lang_name}.
Include SPECIFIC details like the title, theme, or an interesting chapter.
Sound conversational and genuine, like telling a friend. Use at most ONE exclamation point.

Return ONLY the commentary text in {lang_name}, nothing else."""


def get_character_ready_commentary_prompt(
    name: str,
    role: str,
    traits: str,
    motivation: str,
    background: str,
    lang_name: str,
    style_instruction: str
) -> str:
    """
    Generate commentary prompt when a character is ready.

    Args:
        name: Character name
        role: Character role
        traits: Formatted personality traits
        motivation: Character motivation
        background: Character background
        lang_name: Language name
        style_instruction: Language-specific style instruction

    Returns:
        Formatted prompt string
    """
    return f"""A new character is ready!

Name: {name}
Role: {role}
Personality: {traits}
Motivation: {motivation}
Background: {background}

=== LANGUAGE REQUIREMENT ===
IMPORTANT: You MUST respond in {lang_name}.
{style_instruction}

Generate a brief (2-3 sentences) natural introduction to this character IN {lang_name}.
Include SPECIFIC traits and motivation, but sound conversational and genuine.
Use at most ONE exclamation point. Kid-friendly but not over-the-top.

Return ONLY the commentary text in {lang_name}, nothing else."""


def get_character_ready_fallback_prompt(
    name: str,
    role: str,
    lang_name: str
) -> str:
    """
    Generate fallback commentary prompt when character details unavailable.

    Args:
        name: Character name
        role: Character role
        lang_name: Language name

    Returns:
        Formatted prompt string
    """
    return f"""A new character just joined: {name} ({role})

=== LANGUAGE REQUIREMENT ===
IMPORTANT: You MUST respond in {lang_name}.

Generate a brief (1-2 sentences) natural introduction IN {lang_name}.
Sound conversational. Use at most ONE exclamation point.

Return ONLY the commentary text in {lang_name}, nothing else."""


def get_chapter_ready_commentary_prompt(
    chapter_number: int,
    title: str,
    synopsis: str,
    educational: str,
    vocab: str,
    word_count: str,
    lang_name: str,
    style_instruction: str
) -> str:
    """
    Generate commentary prompt when a chapter is ready.

    Args:
        chapter_number: Chapter number
        title: Chapter title
        synopsis: Chapter synopsis preview
        educational: Formatted educational points
        vocab: Formatted vocabulary words
        word_count: Word count string
        lang_name: Language name
        style_instruction: Language-specific style instruction

    Returns:
        Formatted prompt string
    """
    return f"""Chapter {chapter_number} is ready: "{title}"

Synopsis: {synopsis}{educational}{vocab}
Word count: {word_count}

=== LANGUAGE REQUIREMENT ===
IMPORTANT: You MUST respond in {lang_name}.
{style_instruction}

Generate a brief (2-3 sentences) natural, engaging comment about this chapter IN {lang_name}.
Include SPECIFIC details from the synopsis or educational points.
Sound conversational and genuine. Use at most ONE exclamation point.

Return ONLY the commentary text in {lang_name}, nothing else."""


def get_chapter_ready_fallback_prompt(
    chapter_number: int,
    title: str,
    lang_name: str
) -> str:
    """
    Generate fallback commentary prompt when chapter details unavailable.

    Args:
        chapter_number: Chapter number
        title: Chapter title
        lang_name: Language name

    Returns:
        Formatted prompt string
    """
    return f"""Chapter {chapter_number} is ready: "{title}"

=== LANGUAGE REQUIREMENT ===
IMPORTANT: You MUST respond in {lang_name}.

Generate a brief (1-2 sentences) natural comment IN {lang_name}.
Sound conversational. Use at most ONE exclamation point.

Return ONLY the commentary text in {lang_name}, nothing else."""


def get_fallback_commentary_prompt(lang_name: str) -> str:
    """
    Generate generic fallback commentary prompt.

    Args:
        lang_name: Language name

    Returns:
        Formatted prompt string
    """
    return f"""Something interesting just happened in your story.

=== LANGUAGE REQUIREMENT ===
IMPORTANT: You MUST respond in {lang_name}.

Generate a brief (1-2 sentences) natural, warm comment in {lang_name}."""
