"""
Continuity Editor's Plot Threads Review Prompt

The Continuity Editor tracks plot elements, identifies new threads,
checks resolutions, and flags elements at risk of becoming dangling.
"""

import json
from typing import Dict, Any, List, Optional


def get_continuity_threads_prompt(
    chapter_number: int,
    total_chapters: int,
    story_title: str,
    story_theme: str,
    chapter_content: str,
    existing_chars_context: str,
    full_plot_elements: List[Dict[str, Any]],
    previous_chapter_content: Optional[str] = None
) -> str:
    """
    Generate the Continuity Editor's plot threads review prompt.

    The Continuity Editor evaluates:
    - New plot elements introduced in the chapter
    - Resolution of existing plot elements
    - Elements at risk of becoming dangling threads
    - Character knowledge consistency

    Args:
        chapter_number: Current chapter being reviewed
        total_chapters: Total chapters in the story
        story_title: Title of the story
        story_theme: Core theme of the story
        chapter_content: Full chapter text to review
        existing_chars_context: Formatted string of existing characters
        full_plot_elements: Complete plot elements with status and setup/resolution
        previous_chapter_content: Previous chapter for thread tracking

    Returns:
        Formatted prompt string for Continuity's review
    """
    is_final_chapter = chapter_number == total_chapters

    # Previous chapter for thread tracking
    prev_chapter_section = ""
    if previous_chapter_content:
        prev_chapter_section = f"""
            === PREVIOUS CHAPTER CONTENT (for thread tracking) ===
            {previous_chapter_content}
            """

    return f"""ROUND TABLE REVIEW - Chapter {chapter_number} of {total_chapters}
            You are the Continuity Editor, tracking plot threads for this story.

            === STORY CONTEXT ===
            Story Title: {story_title}
            Story Theme: {story_theme}
            This is Chapter {chapter_number} of {total_chapters} total chapters.
            Is this the FINAL chapter? {"🚨 YES - ALL major threads must be resolved!" if is_final_chapter else "No"}
            {prev_chapter_section}
{existing_chars_context}
            === CHAPTER WRITTEN (COMPLETE) ===
            {chapter_content}

            === FULL PLOT ELEMENTS BEING TRACKED ===
            {json.dumps(full_plot_elements, indent=2) if full_plot_elements else "No plot elements tracked yet."}

            === YOUR DOMAIN: CONTINUITY & PLOT THREADS ===
            Review for:
            1. IDENTIFY new plot elements introduced in this chapter
               - Mysteries, objects, conflicts, promises, relationships, secrets
            2. CHECK if any existing plot elements were resolved (match against setup_text)
            3. FLAG elements at risk of becoming dangling threads
            4. VERIFY character knowledge consistency (compare with previous chapter)

            === VERDICT RULES (FOLLOW STRICTLY) ===
            BLOCK if ANY of these are true:
            - FINAL CHAPTER: Any major plot element (importance="major") remains unresolved
            - Critical continuity error (character knows something they shouldn't)
            - Major setup with no clear path to payoff
            - Previous chapter events contradicted

            CONCERN if:
            - Plot element introduced 2+ chapters ago with no progress
            - Minor inconsistency that should be addressed
            - Thread at risk of being forgotten
            - Setup_text not adequately followed up

            APPROVE if:
            - All active threads are being tracked
            - New elements are properly set up
            - No continuity errors detected
            - Previous chapter's events respected

            {"🚨 FINAL CHAPTER ALERT: This is the last chapter. ALL major plot elements MUST be resolved!" if is_final_chapter else ""}

            Output JSON:
            {{
                "agent": "Continuity",
                "domain": "continuity",
                "verdict": "approve" or "concern" or "block",
                "praise": "What threads are well-tracked...",
                "concern": "What threads risk being forgotten...",
                "suggestion": "How to ensure resolution...",
                "plot_elements_new": [
                    {{"name": "element name", "type": "mystery|object|conflict|promise|relationship|secret", "setup_text": "brief setup..."}}
                ],
                "plot_elements_resolved": [
                    {{"name": "element name", "resolution_text": "how it was resolved..."}}
                ],
                "plot_elements_at_risk": [
                    {{"name": "element name", "introduced_chapter": 2, "risk": "description of risk..."}}
                ]
            }}"""
