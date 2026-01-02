"""
Chapter Polish Prompt

Final polish pass to apply reviewer suggestions (especially line edits
from Benjamin the Line Editor) while preserving the chapter's soul.
"""

from typing import Dict, Any


def get_polish_chapter_prompt(
    chapter_number: int,
    language_name: str,
    prose_instruction: str,
    chapter_content: str,
    suggestions_text: str
) -> str:
    """
    Generate the chapter polish prompt for NarrativeAgent.

    The polish prompt applies subtle refinements suggested by reviewers
    without changing plot, characters, or educational content.

    Args:
        chapter_number: Chapter being polished
        language_name: Name of the target language (e.g., "Norwegian Bokmål")
        prose_instruction: Language-specific prose instructions
        chapter_content: The approved chapter content to polish
        suggestions_text: Formatted suggestions from reviewers

    Returns:
        Formatted prompt string for chapter polish
    """
    return f"""POLISH PASS - Chapter {chapter_number}

            === CRITICAL: LANGUAGE REQUIREMENT ===
            LANGUAGE: You MUST output the polished chapter in {language_name}.
            The chapter is written in {language_name} - DO NOT translate it to any other language.
            {prose_instruction}

            Your chapter has been APPROVED by the Round Table, but reviewers have provided
            suggestions for refinement. Apply these improvements while preserving the chapter's soul.

            APPROVED CHAPTER:
            {chapter_content}

            === REVIEWER SUGGESTIONS ===
            {suggestions_text}

            === POLISH INSTRUCTIONS ===
            1. Apply SPECIFIC line edits suggested (especially from Benjamin the Line Editor)
            2. Fix any redundancies mentioned ("nodded his head" → "nodded")
            3. Improve sentence rhythm where suggested
            4. Enhance "show don't tell" where flagged
            5. Keep sensory details and immersive elements
            6. DO NOT change plot, characters, or educational content
            7. DO NOT add new scenes or remove existing ones
            8. MAINTAIN the original {language_name} language - DO NOT translate

            Output the COMPLETE polished chapter in {language_name} (just the narrative text, not JSON).
            The changes should be subtle refinements, not rewrites or translations."""
