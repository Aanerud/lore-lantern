"""
Chapter Writing Prompt

Main chapter drafting prompt for NarrativeAgent (Nnedi Okofor persona).
Generates age-appropriate narrative with sensory grounding and educational content.
"""

from typing import List, Optional


def get_write_chapter_prompt(
    chapter_number: int,
    chapter_title: str,
    chapter_synopsis: str,
    story_title: str,
    character_names: List[str],
    educational_points: List[str],
    target_age: int,
    word_limits: str,
    language_instruction: str,
    sensory_guidance: str,
    story_so_far_context: str = "",
    chapter_1_context: str = "",
    user_requests_section: str = "",
    user_style_requests_section: str = ""
) -> str:
    """
    Generate the main chapter writing prompt for NarrativeAgent.

    This prompt instructs Nnedi to write engaging, age-appropriate narrative
    with proper sensory grounding and educational content integration.

    Args:
        chapter_number: Chapter number to write
        chapter_title: Title of the chapter
        chapter_synopsis: Synopsis/blueprint for the chapter
        story_title: Title of the story
        character_names: List of character names to feature
        educational_points: Educational content to integrate
        target_age: Target reader age
        word_limits: Age-appropriate word/sentence limits
        language_instruction: Language-specific prose instructions
        sensory_guidance: Age-specific sensory writing guidance
        story_so_far_context: Summary of previous chapters (for ch 2+)
        chapter_1_context: Special first chapter requirements (for ch 1 only)
        user_requests_section: User-specified requests to incorporate
        user_style_requests_section: User style preferences from setup

    Returns:
        Formatted prompt string for chapter writing
    """
    return f"""Write Chapter {chapter_number}: "{chapter_title}"
{language_instruction}
        Story: {story_title}
        Synopsis: {chapter_synopsis}
        Characters: {', '.join(character_names)}
        Educational points to integrate: {', '.join(educational_points)}
{story_so_far_context}
{chapter_1_context}
{user_requests_section}
{user_style_requests_section}
        === SENSORY GROUNDING REQUIREMENTS (CRITICAL) ===
        Every paragraph MUST include at least ONE sensory detail:
        - VISUAL: Colors, light, textures, shapes, movement
        - AUDITORY: Sounds, voice quality, silence, echoes
        - TACTILE: Temperature, texture, pressure, physical sensations
        - OLFACTORY: Smells (especially in meals, nature, danger scenes)

        POV IMMERSION: Write from INSIDE the character's experience:
        - BAD: "The wind was cold." (external observation)
        - GOOD: "Harald felt the cold wind bite his cheeks." (inside experience)

        Show emotions through PHYSICAL sensations:
        - Fear: dry mouth, trembling hands, racing heart
        - Excitement: tingling skin, quickened breath
        - Sadness: heavy chest, tight throat

{sensory_guidance}
        Write engaging narrative for age {target_age}:
        - {word_limits}
        - Age-appropriate language (simple words for younger readers)
        - Include 2-3 new vocabulary words in context
        - Weave in educational content naturally
        - End with a hook for the next chapter

        Output valid JSON with all required fields."""


def get_write_chapter_json_schema(
    chapter_number: int,
    chapter_title: str,
    chapter_synopsis: str
) -> str:
    """
    Generate the JSON output schema for chapter writing.

    This schema is appended to the write prompt to guide the output format.

    Args:
        chapter_number: Chapter number
        chapter_title: Title of the chapter
        chapter_synopsis: Synopsis of the chapter

    Returns:
        JSON schema template string
    """
    return f"""

            {{
                "number": {chapter_number},
                "title": "{chapter_title}",
                "synopsis": "{chapter_synopsis}",
                "content": "The full narrative text...",
                "characters_featured": ["names"],
                "educational_points": ["points covered"],
                "vocabulary_words": [
                    {{
                        "word": "navigator",
                        "definition": "person who guides a ship",
                        "age_appropriate_level": 8,
                        "context_in_story": "sentence where it appears"
                    }}
                ],
                "facts": [
                    {{
                        "fact": "historical claim",
                        "verified": false,
                        "confidence": 0.8
                    }}
                ],
                "word_count": 800,
                "reading_time_minutes": 5
            }}"""
