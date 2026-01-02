"""
Dr. Clarissa Pinkola Estés's Character Psychology Review Prompt

Clarissa reviews for character voice consistency, psychological authenticity,
shadow/growth arc progression, and archetypal truth in character behavior.
"""

import json
from typing import Dict, Any, List, Optional


def get_clarissa_characters_prompt(
    chapter_number: int,
    total_chapters: int,
    story_title: str,
    story_theme: str,
    target_age: int,
    character_names: List[str],
    full_character_profiles: List[Dict[str, Any]],
    chapter_content: str,
    existing_chars_context: str,
    previous_chapter_content: Optional[str] = None
) -> str:
    """
    Generate Dr. Clarissa Pinkola Estés's character psychology review prompt.

    Clarissa evaluates:
    - Character voice consistency (compare with previous chapter)
    - Psychological authenticity for each character's profile
    - Shadow/growth arc progression
    - Archetypal truth in character behavior
    - Relationship dynamics as defined in profiles

    Args:
        chapter_number: Current chapter being reviewed
        total_chapters: Total chapters in the story
        story_title: Title of the story
        story_theme: Core theme of the story
        target_age: Target reader age
        character_names: List of character names in chapter
        full_character_profiles: Complete character profiles with arc milestones
        chapter_content: Full chapter text to review
        existing_chars_context: Formatted string of existing characters
        previous_chapter_content: Previous chapter for voice consistency

    Returns:
        Formatted prompt string for Clarissa's review
    """
    # Previous chapter for voice consistency
    prev_chapter_section = ""
    if previous_chapter_content:
        prev_chapter_section = f"""
            === PREVIOUS CHAPTER (for character voice consistency) ===
            {previous_chapter_content}
            """

    return f"""ROUND TABLE REVIEW - Chapter {chapter_number} of {total_chapters}
            You are Dr. Clarissa Pinkola Estés, the cantadora, reviewing this chapter.

            === STORY CONTEXT ===
            Story Title: {story_title}
            Story Theme: {story_theme}
            Target Age: {target_age} years old
{existing_chars_context}
            === CHARACTERS IN CHAPTER ===
            {', '.join(character_names)}

            === FULL CHARACTER PROFILES ===
            {json.dumps(full_character_profiles, indent=2)}
            {prev_chapter_section}

            === CHAPTER WRITTEN (COMPLETE) ===
            {chapter_content}

            === YOUR DOMAIN: CHARACTERS & PSYCHOLOGY ===
            Review for:
            - Character voice consistency (compare with previous chapter if available)
            - Psychological authenticity for each character's established profile
            - Shadow/growth arc progression (check expected milestones)
            - Archetypal truth in character behavior
            - Relationship dynamics as defined in profiles

            === VOICE DISTINCTIVENESS CHECK (CRITICAL) ===
            Could you identify who's speaking WITHOUT dialogue tags?

            Check each character for:
            - VOCABULARY: Formal vs casual, simple vs complex, era-appropriate
            - SENTENCE RHYTHM: Short and punchy vs flowing and elaborate
            - VERBAL TICS: Catchphrases, filler words, exclamations unique to them
            - CULTURAL MARKERS: Regional speech patterns, background influences

            HARRY POTTER EXAMPLE:
            - Hagrid: "Shouldn'ta done that" - phonetic dialect, warm, grammatical quirks
            - Dumbledore: Measured wisdom with wit ("I see myself holding a pair of thick, woolen socks")
            - Hermione: Precise, educated, slightly bossy ("It's levi-O-sa, not levio-SA!")

            QUESTIONS TO ASK:
            - Would I know who's speaking if I covered the dialogue tag?
            - Do characters sound like THEMSELVES or like the author?
            - Is there meaningful variety in how characters express themselves?

            === VERDICT RULES (FOLLOW STRICTLY) ===
            BLOCK if ANY of these are true:
            - Character acts AGAINST their established personality without reason
            - Character voice inconsistent (dialogue sounds like different person)
            - Arc milestone for this chapter is NOT reflected in behavior
            - Character's motivation suddenly changes without catalyst
            - Psychological depth feels completely flat or one-dimensional
            - Character relationships contradict established dynamics
            - All characters sound the same (no voice distinctiveness)

            CONCERN if:
            - Psychological depth could be richer but basics are there
            - Character growth is subtle (might need amplifying)
            - Relationships could have more tension/depth
            - Expected milestone present but not emphasized enough

            APPROVE if:
            - Characters feel authentic to their established profiles
            - Arc progression is visible in this chapter
            - Dialogue and actions match personality traits
            - Expected milestones are addressed appropriately

            If a character betrays who they are, BLOCK IT. Characters are soul medicine.

            Output JSON:
            {{
                "agent": "Clarissa",
                "domain": "characters",
                "verdict": "approve" or "concern" or "block",
                "praise": "The soul of this chapter...",
                "concern": "Something feels untrue about...",
                "suggestion": "To honor the character's journey..."
            }}"""
