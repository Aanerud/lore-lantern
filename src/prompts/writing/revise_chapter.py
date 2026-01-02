"""
Chapter Revision Prompt

Nnedi revises chapters based on Round Table feedback while maintaining
the story's emotional truth and language consistency.
"""

from typing import Dict, Any


def get_revise_chapter_prompt(
    chapter_number: int,
    chapter_title: str,
    language_name: str,
    prose_instruction: str,
    original_content: str,
    revision_guidance: str
) -> str:
    """
    Generate the chapter revision prompt for NarrativeAgent.

    The revision prompt instructs Nnedi to address Round Table feedback
    while preserving the chapter's soul and language.

    Args:
        chapter_number: Chapter being revised
        chapter_title: Title of the chapter
        language_name: Name of the target language (e.g., "Norwegian Bokmål")
        prose_instruction: Language-specific prose instructions
        original_content: The original chapter content to revise
        revision_guidance: Compiled guidance from Round Table reviews

    Returns:
        Formatted prompt string for chapter revision
    """
    return f"""CHAPTER REVISION - Chapter {chapter_number}: "{chapter_title}"

            === CRITICAL: LANGUAGE REQUIREMENT ===
            LANGUAGE: You MUST output the revised chapter in {language_name}.
            The chapter is written in {language_name} - DO NOT translate it to any other language.
            {prose_instruction}

            Your colleagues at the Round Table have reviewed your draft and provided feedback.
            Revise your chapter while maintaining its soul.

            ORIGINAL CHAPTER (FULL TEXT - ADDRESS ALL ISSUES):
            {original_content}

            {revision_guidance}

            IMPORTANT:
            - Address ALL concerns raised
            - Maintain the story's emotional truth
            - Keep what works well
            - Preserve educational elements
            - Don't change the fundamental plot direction
            - MAINTAIN the original {language_name} language - DO NOT translate

            Output the COMPLETE revised chapter content in {language_name} (just the narrative text, not JSON)."""
