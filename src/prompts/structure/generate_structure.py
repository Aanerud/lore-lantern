"""
Story Structure Generation Prompt

Creates the initial story outline with chapter synopses, character roster,
educational goals, and character arc milestones.
"""

from typing import List, Optional


def get_generate_structure_prompt(
    story_prompt: str,
    difficulty: str,
    target_age: int,
    educational_focus: str,
    themes: str,
    scary_level: str,
    dialogue_context: str,
    user_style_requests: str,
    is_historical: bool
) -> str:
    """
    Generate the story structure creation prompt for StructureAgent.

    This prompt creates a complete story outline including:
    - Chapter synopses following 4-part structure
    - Character roster with arc milestones
    - Educational goals
    - Character development milestones per chapter

    Args:
        story_prompt: User's story request
        difficulty: Story difficulty level (easy/medium/hard)
        target_age: Target reader age
        educational_focus: Educational focus area
        themes: Comma-separated theme list
        scary_level: Scary content level
        dialogue_context: Context from user dialogue
        user_style_requests: Formatted user style preferences
        is_historical: Whether this is a historical story

    Returns:
        Formatted prompt string for structure generation
    """
    historical_note = 'For this HISTORICAL story, include family members, allies, advisors, and rivals.' if is_historical else ''

    return f"""Create a {difficulty} story outline for: "{story_prompt}"

            TARGET AUDIENCE: {target_age}-year-old child
            - Use age-appropriate vocabulary and sentence complexity
            - Themes and content suitable for age {target_age}

            User preferences:
            - Educational focus: {educational_focus}
            - Themes: {themes}
            - Scary level: {scary_level}
            - Dialogue context: {dialogue_context}
            {user_style_requests}

            IMPORTANT - CHARACTER PLANNING:
            Determine how many characters this story needs based on complexity:
            - Simple stories (single hero journey): 3-5 characters
            - Medium stories (team adventures): 5-8 characters
            - Complex stories (historical epics, multiple plotlines): 8-12 characters

            {historical_note}

            For MAJOR characters (protagonist, main antagonist, key mentor/ally):
            Plan character arc milestones showing how they grow across chapters.
            Example for a young protagonist:
            - Chapter 1: "Young, curious child discovers a mysterious secret"
            - Chapter 3: "Learns that courage requires responsibility"
            - Chapter 5: "Becomes a leader who balances boldness with wisdom"

            For each chapter, specify character development milestones (what characters learn/experience).

            SYNOPSIS WRITING REQUIREMENTS (CRITICAL):
            Write professional-quality chapter synopses following industry standards (Harry Potter, Percy Jackson quality).

            LENGTH & FORMAT:
            - 150-200 words per synopsis
            - Third person, present tense
            - Active voice ("Maya solves the puzzle" not "The puzzle is solved")
            - Narrative paragraph style (NOT bullet points or lists)

            REQUIRED 4-PART STRUCTURE:
            1. OPENING (1-2 sentences):
               - Establish protagonist's immediate situation
               - Set the scene and context
               Example: "Maya arrives at the old factory on the hill, curious about the strange lights."

            2. DEVELOPMENT (4-6 sentences):
               - Cover 3-4 key plot beats in chronological order
               - Include character reactions and emotional states
               - Show conflicts and obstacles encountered
               - Weave educational moments naturally into narrative (don't list them separately!)
               - Use cause-effect connections ("When X happens, Y results")
               Example: "Old Mr. Chen warns Maya about the factory's secrets but quickly realizes her curiosity cannot be contained. Through a tense exploration, Maya learns that some mysteries require patience to unravel safely."

            3. CLIMAX (1-2 sentences):
               - The chapter's peak moment
               - A decision, discovery, or confrontation
               Example: "Maya discovers the hidden workshop where magical toys come to life."

            4. RESOLUTION/HOOK (1-2 sentences):
               - How the chapter concludes
               - Setup for next chapter
               Example: "Maya sneaks back home at twilight, determined to return tomorrow with her best friend."

            QUALITY STANDARDS:
            ✓ Use character names (not excessive "he/she/they")
            ✓ Include emotional beats ("terrified", "determined", "conflicted", "amazed")
            ✓ Reveal outcomes (this is NOT a teaser - show what happens, including twists)
            ✓ Educational content integrated naturally into plot
            ✓ Clear chronological progression
            ✓ Engaging tone without being overdramatic
            ✓ Identify ALL characters who appear (major and supporting)

            AVOID:
            ✗ Dialogue quotes
            ✗ Minor subplot details
            ✗ Scene-by-scene breakdowns
            ✗ Vague summaries ("stuff happens", "they talk")
            ✗ Bullet points or lists

            IMPORTANT: Each synopsis must identify ALL characters who appear in that chapter, including minor characters who may not be in the initial characters_needed list. This ensures complete character roster for later agents.

            Output valid JSON matching this format:
            {{
                "title": "Story Title",
                "theme": "main theme",
                "chapters": [
                    {{
                        "number": 1,
                        "title": "Chapter Title",
                        "synopsis": "Maya arrives at the old toy factory on Maple Hill, curious about the strange lights her grandmother mentioned in old stories. Mr. Chen, the elderly caretaker, warns her about the building's mysteries but quickly realizes her curiosity cannot be contained, leading to a tense exploration where Maya discovers hidden passages behind dusty shelves. Through careful observation, Maya learns that some secrets require patience and respect to uncover safely—a lesson she struggles to accept given her adventurous spirit. As evening approaches, Maya discovers a workshop where handcrafted toys seem to move on their own when no one is watching. The chapter closes with Maya sneaking home at twilight, determined to return tomorrow with her best friend Lily to investigate further.",
                        "characters_featured": ["Maya", "Mr. Chen"],
                        "educational_points": ["Respecting history and heritage", "Patience in problem-solving"],
                        "facts_to_verify": ["Traditional toy-making craftsmanship"],
                        "character_development_milestones": {{
                            "Maya": "First encounter with real magic and mystery",
                            "Mr. Chen": "Begins to trust Maya with the factory's secrets"
                        }}
                    }}
                ],
                "characters_needed": [
                    {{
                        "name": "Maya",
                        "role": "protagonist",
                        "importance": "major",
                        "arc_milestones": {{
                            "1": "Curious and impulsive explorer",
                            "3": "Learns responsibility for her discoveries",
                            "5": "Wise guardian of magical secrets"
                        }}
                    }},
                    {{
                        "name": "Mr. Chen",
                        "role": "mentor",
                        "importance": "major"
                    }},
                    {{
                        "name": "Lily",
                        "role": "sidekick",
                        "importance": "supporting"
                    }}
                ],
                "educational_goals": [
                    {{
                        "concept": "Heritage Preservation",
                        "description": "The importance of preserving craftsmanship and traditions",
                        "age_appropriate": true
                    }}
                ],
                "estimated_reading_time_minutes": 20
            }}"""
