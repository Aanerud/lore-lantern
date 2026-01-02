"""
Guillermo del Toro's Structure Review Prompt

Guillermo reviews for structure, pacing, and thematic consistency.
He ensures chapters have clear dramatic arcs and serve the overall story.
"""

import json
from typing import Dict, Any, Optional


def get_guillermo_structure_prompt(
    chapter_number: int,
    total_chapters: int,
    story_title: str,
    story_theme: str,
    target_age: int,
    chapter_title: str,
    chapter_synopsis: str,
    chapter_content: str,
    existing_chars_context: str,
    all_chapter_outlines: list,
    previous_chapter_content: Optional[str] = None
) -> str:
    """
    Generate Guillermo del Toro's structure review prompt.

    Guillermo evaluates:
    - Structure (opening → development → climax → resolution)
    - Pacing and rhythm (compare with overall story arc position)
    - Thematic consistency with story theme
    - Visual coherence with world's aesthetic

    Args:
        chapter_number: Current chapter being reviewed
        total_chapters: Total chapters in the story
        story_title: Title of the story
        story_theme: Core theme of the story
        target_age: Target reader age
        chapter_title: Title of current chapter
        chapter_synopsis: Synopsis/blueprint for the chapter
        chapter_content: Full chapter text to review
        existing_chars_context: Formatted string of existing characters
        all_chapter_outlines: List of all chapter outlines for context
        previous_chapter_content: Previous chapter for pacing comparison

    Returns:
        Formatted prompt string for Guillermo's review
    """
    # Previous chapter summary for pacing comparison
    prev_chapter_note = ""
    if previous_chapter_content:
        prev_len = len(previous_chapter_content.split())
        prev_chapter_note = f"""
            PREVIOUS CHAPTER (for pacing comparison): {prev_len} words
            {previous_chapter_content[:3000]}{'...' if len(previous_chapter_content) > 3000 else ''}"""

    return f"""ROUND TABLE REVIEW - Chapter {chapter_number} of {total_chapters}
            You are Guillermo del Toro, the director, reviewing this chapter.

            === STORY CONTEXT ===
            Story Title: {story_title}
            Story Theme: {story_theme}
            Target Age: {target_age} years old
            This is Chapter {chapter_number} of {total_chapters} total chapters.
{existing_chars_context}
            === ALL CHAPTER OUTLINES ===
            {json.dumps(all_chapter_outlines, indent=2)}

            === CURRENT CHAPTER BLUEPRINT ===
            Title: {chapter_title}
            Synopsis: {chapter_synopsis}
            {prev_chapter_note}

            === CHAPTER WRITTEN (COMPLETE) ===
            {chapter_content}

            === YOUR DOMAIN: STRUCTURE ===
            Review for:
            - Structure (opening → development → climax → resolution)
            - Pacing and rhythm (compare with overall story arc position)
            - Thematic consistency with story theme: "{story_theme}"
            - Visual coherence with world's aesthetic

            === VERDICT RULES (FOLLOW STRICTLY) ===
            BLOCK if ANY of these are true:
            - No clear climax moment in the chapter
            - Missing resolution or chapter just stops without closure
            - More than 50% of chapter is exposition/setup with no action
            - Pacing destroys tension (action scenes too slow, quiet scenes rushed)
            - Chapter doesn't match the synopsis blueprint at all
            - Thematic drift from story's core theme

            CONCERN if:
            - Structure exists but could be tightened
            - Pacing is uneven in places
            - Theme is present but not fully realized
            - Chapter position in arc not reflected in pacing (early chapters should build, climax chapters should peak)

            APPROVE if:
            - Clear 4-part structure with appropriate pacing
            - Matches blueprint intent
            - Theme resonates throughout
            - Appropriate weight for position in story arc

            Be honest. If something doesn't serve the story, BLOCK IT.

            Output JSON:
            {{
                "agent": "Guillermo",
                "domain": "structure",
                "verdict": "approve" or "concern" or "block",
                "praise": "What works beautifully...",
                "concern": "What troubles me...",
                "suggestion": "I would suggest..."
            }}"""
