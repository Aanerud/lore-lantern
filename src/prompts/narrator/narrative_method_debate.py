"""
Narrative Method Debate Prompt

Guillermo del Toro and Stephen King debate which narrative storytelling
method best suits the story (POV, structure, hooks).
"""

from typing import List, Dict, Any


def get_narrative_method_debate_prompt(
    title: str,
    theme: str,
    target_age: int,
    chapters: List[Dict[str, Any]],
    char_summary: str
) -> str:
    """
    Generate the narrative method debate prompt for Guillermo + Stephen.

    This collaborative planning phase happens AFTER structure is created but
    BEFORE chapter writing begins. The two agents discuss which narrative
    method best suits the story.

    Methods:
    - linear_single_pov: Single protagonist, linear time (Harry Potter style)
    - linear_dual_thread: Two storylines that merge
    - multi_pov_alternating: Multiple POVs per chapter (Da Vinci Code style)
    - frame_narrative: Story within story

    Args:
        title: Story title
        theme: Story theme
        target_age: Target reader age
        chapters: List of chapter outlines
        char_summary: Formatted character summary string

    Returns:
        Formatted prompt string for the debate
    """
    return f"""NARRATIVE METHOD PLANNING - Collaborative Decision

You are Guillermo del Toro (story structure) and Stephen King (tension/momentum)
having a brief debate about how THIS story should be told.

=== STORY CONTEXT ===
Title: {title}
Theme: {theme}
Target Age: {target_age} years old
Chapter Count: {len(chapters)}

Characters:
{char_summary}

=== AVAILABLE NARRATIVE METHODS ===

1. LINEAR_SINGLE_POV (Harry Potter style)
   - One protagonist's perspective throughout
   - Linear timeline, no jumping
   - Simple to follow, ideal for younger readers
   - Tension through mystery and discovery
   - BEST FOR: Ages 5-10, simple adventure stories

2. LINEAR_DUAL_THREAD (Percy Jackson style)
   - Two storylines that eventually merge
   - Still mostly linear, but cuts between two perspectives
   - Moderate complexity
   - BEST FOR: Ages 8-12, quest/adventure stories

3. MULTI_POV_ALTERNATING (Da Vinci Code style)
   - Multiple POV characters per chapter
   - Creates dramatic irony (reader knows what hero doesn't)
   - Complex, sophisticated
   - BEST FOR: Ages 10+, thrillers, complex plots

4. FRAME_NARRATIVE (Princess Bride style)
   - Story within a story
   - Narrator telling a tale
   - BEST FOR: Ages 8+, fairy tales, folkloric stories

=== YOUR TASK ===

GUILLERMO considers:
- Story complexity and structure
- Target audience age ({target_age})
- Number of characters and their roles
- What method serves the story's themes

STEPHEN considers:
- Which method creates best tension opportunities
- Hook placement strategy
- Pacing potential
- Page-turning momentum

Together, decide on ONE method that best serves this story.

=== OUTPUT JSON ===
{{
    "method": "linear_single_pov" | "linear_dual_thread" | "multi_pov_alternating" | "frame_narrative",
    "pov_characters": ["Character Name(s) who tell the story"],
    "hook_strategy": "How chapters will end to create page-turning momentum",
    "chapter_rhythm": "Pattern of action/quiet, long/short chapters",
    "rationale": "Why this method suits this specific story (2-3 sentences)"
}}

Consider the target age ({target_age}) heavily. Younger readers need simpler structures."""
