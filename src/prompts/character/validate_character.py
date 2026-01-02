"""
Character Validation Prompt

Bill Nye validates characters for historical accuracy and duplicate detection.
"""

from typing import Optional


def get_validate_character_prompt(
    character_name: str,
    character_card: str,
    story_context: str,
    existing_chars_summary: str
) -> str:
    """
    Generate Bill Nye's character validation prompt.

    Bill validates characters for:
    - Duplicate detection (same person, different name)
    - Historical vs fictional classification
    - Timeline accuracy for historical figures
    - Anachronism detection

    Args:
        character_name: Name of character being validated
        character_card: Formatted character card for review
        story_context: Brief context about story setting/era
        existing_chars_summary: Formatted existing characters for dedupe

    Returns:
        Formatted prompt string for character validation
    """
    return f"""CHARACTER VALIDATION - Bill Nye Review

You are Bill Nye, reviewing a character card for DUPLICATES and historical accuracy.

=== STORY CONTEXT ===
{story_context}

=== CHARACTER TO VALIDATE ===
{character_card}
{existing_chars_summary}

=== YOUR TASK ===
Analyze this character and determine:

1. DUPLICATE CHECK (if existing characters provided):
   Is this character the SAME PERSON as an existing character?
   - Same role + similar background = likely duplicate
   - "Jarl Eirik the Bold" and "Jarl Eirik Ironbrow" with same role = SAME PERSON
   - Different titles for same person (King Harald = Harald Fairhair) = SAME PERSON
   - BUT: Two different Jarls with different backgrounds = DIFFERENT PEOPLE (allowed!)

2. IS THIS CHARACTER HISTORICAL OR FICTIONAL?
   - HISTORICAL: A real person from history (e.g., Harald Hårfagre, Halfdan the Black)
   - FICTIONAL: A made-up character for this story (e.g., "Skald Orm", "Jarl Eirik Ironbrow")

3. IF HISTORICAL - TIMELINE CHECK:
   For real historical figures, verify they belong in the story's era:

   HARALD HÅRFAGRE ERA (c. 850-930 CE):
   ✅ VALID contemporaries:
   - Halfdan the Black (Harald's father, d. ~860)
   - Queen Ragnhild (legendary mother)
   - Gyda Eiriksdottir (legendary, inspired unification)
   - Various Jarls of the period

   ❌ INVALID - WRONG GENERATION:
   - Eric/Eirik Bloodaxe (Harald's SON, born ~895, king after Harald)
   - Haakon the Good (Harald's son)
   - Any of Harald's children as contemporaries/rivals

4. SUGGESTIONS:
   If duplicate or timeline wrong, suggest corrections.

Output ONLY valid JSON:
{{
    "character_name": "{character_name}",
    "is_duplicate": true or false,
    "duplicate_of": "Name of existing character if duplicate, null otherwise",
    "duplicate_reason": "Why you think they are the same person, null if not duplicate",
    "is_historical": true or false,
    "historical_identity": "Name of real person if historical, null if fictional",
    "timeline_valid": true or false or null (null if fictional),
    "era_check": "Explanation of timeline validity",
    "suggestions": ["List of suggestions if any issues found"],
    "verdict": "valid" or "duplicate" or "warning" or "error"
}}

IMPORTANT: Fictional characters with creative epithets (e.g., "Ironbrow", "Storm-Brow")
are ALWAYS valid - they don't need historical verification."""
