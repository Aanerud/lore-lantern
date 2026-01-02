"""
Character Creation Prompt

Creates detailed character profiles with D&D-style skill progression,
Jungian psychological depth, and story arc integration.
"""

import json
from typing import Optional, Dict, Any


def get_create_character_prompt(
    char_name: str,
    char_role: str,
    char_importance: str,
    story_title: str,
    story_theme: str,
    difficulty: str,
    existing_chars_context: str,
    arc_description: str,
    skill_requirement: str,
    min_skills: int,
    arc_milestones: Optional[Dict[str, str]]
) -> str:
    """
    Generate the character creation prompt for CharacterAgent.

    Creates detailed character profiles with:
    - Jungian psychological depth (light/shadow traits)
    - D&D-style skill progression
    - Character arc integration
    - Relationship mapping

    Args:
        char_name: Character name to create
        char_role: Role in the story (protagonist, mentor, etc.)
        char_importance: major/supporting/minor
        story_title: Title of the story
        story_theme: Core theme of the story
        difficulty: Story difficulty level
        existing_chars_context: Formatted existing characters for dedupe
        arc_description: Formatted arc milestones if provided
        skill_requirement: Required skills based on importance
        min_skills: Minimum number of skills required
        arc_milestones: Optional arc milestones dict

    Returns:
        Formatted prompt string for character creation
    """
    arc_milestones_json = json.dumps(arc_milestones) if arc_milestones else "null"

    return f"""Create a detailed character profile for: {char_name}

            Story context:
            - Title: {story_title}
            - Theme: {story_theme}
            - Target age: {difficulty}
            - Character importance: {char_importance}
            {existing_chars_context}
            {arc_description}

            === REQUIRED PROFILE ELEMENTS ===

            1. IDENTITY
               - Full name (use {char_name} or expand with titles/epithets)
               - Age (specific number or descriptor like "elderly", "young")
               - Role: {char_role}

            2. PSYCHOLOGY (Jungian depth)
               - Background story rooted in culture/mythology (minimum 50 words)
               - 3-5 personality traits (MUST mix LIGHT and SHADOW qualities)
               - Psychological motivation (what they NEED, not just want)
               - Current emotional state (NOT "neutral" - use: eager, determined, anxious, hopeful, grieving, conflicted, curious, wary)

            3. PHYSICAL
               - Appearance with symbolic visual elements
               - Distinctive features that reflect inner character

            4. RELATIONSHIPS
               - How they relate to other known characters
               - Relationship dynamics (mentor, rival, friend, protector, etc.)

            5. SKILLS (D&D-STYLE PROGRESSION) - REQUIRED
               This character needs: {skill_requirement}

               Each skill MUST include ALL fields:
               - name: Specific skill (Combat, Social, Knowledge, or Practical category)
               - level: 1-10 (start LOW so there's room to grow)
               - acquired_chapter: 0 (learned before story begins)
               - description: How THIS character uniquely uses this skill

               CONNECT SKILLS TO ARC: Choose skills that CAN GROW as the character develops.

            6. CHARACTER ARC (if arc milestones provided)
               Map how personality/skills will evolve across chapters.

            Output valid JSON matching this structure:
            {{
                "name": "{char_name}",
                "role": "{char_role}",
                "age": 25,
                "background": "Detailed background story of at least 50 words rooted in culture...",
                "personality_traits": ["brave", "impulsive", "secretly fearful", "loyal", "stubborn"],
                "motivation": "Deep psychological need driving this character...",
                "appearance": "Physical description with symbolic elements...",
                "relationships": {{"other_character_name": "relationship description"}},
                "progression": {{
                    "skills_learned": [
                        {{"name": "Leadership", "level": 2, "acquired_chapter": 0, "description": "Natural ability to inspire others"}},
                        {{"name": "Swordsmanship", "level": 3, "acquired_chapter": 0, "description": "Trained since childhood"}},
                        {{"name": "Diplomacy", "level": 1, "acquired_chapter": 0, "description": "Struggles to negotiate"}}
                    ],
                    "personality_evolution": [],
                    "relationship_changes": [],
                    "current_emotional_state": "eager",
                    "chapters_featured": []
                }},
                "character_arc": {arc_milestones_json}
            }}

            CRITICAL: skills_learned array MUST have at least {min_skills} skills!"""
