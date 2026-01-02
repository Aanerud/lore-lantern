"""
Structure Refinement Prompt (V2)

Refines story synopses after Chapter 1 is written, incorporating:
- D&D-style character skills
- User preferences from dialogue
- Actual Chapter 1 content for continuity
"""

import json
from typing import List, Dict, Any


def get_refine_structure_system_prompt() -> str:
    """
    Get the system prompt for structure refinement.

    Returns:
        System prompt string for Foundry chat completion
    """
    return """You are a Story Architect specializing in refining chapter synopses.
You enhance synopses by integrating D&D-style character skills into plot points.
You always output valid JSON following the exact format requested.
Do not include markdown code blocks in your response."""


def get_refine_structure_prompt(
    story_title: str,
    story_theme: str,
    educational_focus: str,
    total_chapters: int,
    chapter_1_content: str,
    character_cards: List[Dict[str, Any]],
    user_prefs_summary: str,
    original_synopses: List[Dict[str, Any]]
) -> str:
    """
    Generate the structure refinement prompt for chapters 2-N.

    This prompt refines synopses by:
    - Building on actual Chapter 1 content
    - Leveraging D&D-style character skills
    - Incorporating user preferences from dialogue

    Args:
        story_title: Title of the story
        story_theme: Core theme of the story
        educational_focus: Educational focus area
        total_chapters: Total number of chapters
        chapter_1_content: Full content of Chapter 1
        character_cards: List of character cards with skills
        user_prefs_summary: Summary of user preferences
        original_synopses: Original synopses for chapters 2-N

    Returns:
        Formatted prompt string for refinement
    """
    return f"""You are refining story synopses based on what actually happened in Chapter 1.

STORY CONTEXT:
- Title: {story_title}
- Theme: {story_theme}
- Educational Focus: {educational_focus}
- Total Chapters: {total_chapters}

CHAPTER 1 (What actually happened - full content):
{chapter_1_content}

CHARACTER CARDS (D&D-style with skills):
{json.dumps(character_cards, indent=2)}

USER PREFERENCES FROM DIALOGUE:
{user_prefs_summary}

ORIGINAL SYNOPSES FOR CHAPTERS 2-{total_chapters}:
{json.dumps(original_synopses, indent=2)}

YOUR TASK:
Rewrite the synopses for chapters 2-{total_chapters} to:

1. BUILD FROM CHAPTER 1 REALITY
   - Reference actual events, discoveries, and emotional beats from Chapter 1
   - Maintain continuity with the established world and character introductions
   - Build naturally on relationships and conflicts established

2. LEVERAGE CHARACTER SKILLS (D&D-style)
   - Each character has specific skills with levels 1-10
   - Create plot points where characters ACTIVELY USE their skills
   - Example: "Harald uses his Battle Strategy (Level 3) to outmaneuver the ambush"
   - Level 1-2: Character struggles but learns
   - Level 3-4: Character shows competence
   - Level 5+: Character excels

3. WEAVE USER PREFERENCES
   - User expressed: {user_prefs_summary}
   - Incorporate these naturally into the story flow
   - Don't force them if they don't fit

4. MAINTAIN SYNOPSIS QUALITY
   - 150-200 words per synopsis
   - Third person, present tense, active voice
   - 4-part structure: Opening → Development → Climax → Resolution/Hook
   - Include character_development_milestones with skill-specific growth

OUTPUT FORMAT (JSON):
{{
    "refined_chapters": [
        {{
            "number": 2,
            "title": "Chapter Title (may update if needed)",
            "synopsis": "Full 150-200 word synopsis...",
            "characters_featured": ["Name1", "Name2"],
            "educational_points": ["Point 1", "Point 2"],
            "facts_to_verify": [],
            "character_development_milestones": {{
                "CharacterName": "Uses SkillName to achieve X, grows toward Y"
            }}
        }}
    ],
    "skills_leveraged": ["Battle Strategy (Harald)", "Diplomacy (Gunnhild)"],
    "refinement_notes": "Brief explanation of major changes"
}}

IMPORTANT:
- Preserve educational content from original synopses
- Keep the overall story arc direction
- ENHANCE and GROUND - don't fundamentally change the story
- Return ONLY valid JSON, no markdown code blocks"""
