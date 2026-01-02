"""
Review Service for Round Table Chapter Review Processing

This module provides utilities for the Round Table review workflow:
- Building shared review context for all reviewer agents
- Compiling revision guidance from reviewer feedback
- Organizing feedback by severity (block, concern, approve)

Architecture:
- Pure functions with no external dependencies (except validation_service)
- Called by coordinator for Round Table review workflow
- Supports multi-agent review pattern

The Round Table review process:
1. All reviewers receive the same context (via build_review_context)
2. Each reviewer provides verdict + feedback
3. Feedback is compiled into actionable guidance (via compile_revision_guidance)
4. Narrative agent revises based on structured guidance
"""

import json
import logging
from typing import Dict, Any, List

from src.services.validation_service import clean_json_output

logger = logging.getLogger(__name__)


def build_review_context(
    story,
    chapter_number: int,
    chapter_content: str,
    chapter_outline
) -> Dict[str, Any]:
    """
    Build shared context dict for all Round Table reviewers.

    Provides full context including:
    - Story theme and structure
    - Previous chapter content (for consistency checking)
    - Educational goals
    - Target age for all reviewers

    Args:
        story: Story object with structure, preferences, and chapters
        chapter_number: Current chapter being reviewed
        chapter_content: Full content of the chapter
        chapter_outline: ChapterOutline object for current chapter

    Returns:
        Context dict with all information reviewers need
    """
    # Get previous chapter content for cross-chapter consistency
    previous_chapter_content = ""
    if chapter_number > 1 and story.chapters:
        prev_chapters = [ch for ch in story.chapters if ch.number == chapter_number - 1]
        if prev_chapters:
            previous_chapter_content = prev_chapters[0].content

    # Get chapter outlines as simplified dicts for JSON serialization
    all_outlines = []
    if story.structure and story.structure.chapters:
        all_outlines = [
            {
                "number": ch.number,
                "title": ch.title,
                "synopsis": ch.synopsis[:500] if len(ch.synopsis) > 500 else ch.synopsis,
                "character_development_milestones": ch.character_development_milestones
            }
            for ch in story.structure.chapters
        ]

    # Get educational goals as simplified dicts
    educational_goals = []
    if story.structure and story.structure.educational_goals:
        educational_goals = [
            {"concept": eg.concept, "description": eg.description}
            for eg in story.structure.educational_goals
        ]

    return {
        "story_title": story.structure.title if story.structure else "Untitled",
        "story_theme": story.structure.theme if story.structure else "Unknown",
        "total_chapters": len(story.structure.chapters) if story.structure else 5,
        "target_age": story.preferences.target_age if story.preferences and story.preferences.target_age else 10,
        "educational_goals": educational_goals,
        "all_chapter_outlines": all_outlines,
        "previous_chapter_content": previous_chapter_content,
        "current_chapter_content": chapter_content,  # No truncation!
        "chapter_number": chapter_number,
        "current_chapter_outline": {
            "title": chapter_outline.title,
            "synopsis": chapter_outline.synopsis,
            "facts_to_verify": chapter_outline.facts_to_verify,
            "character_development_milestones": chapter_outline.character_development_milestones
        }
    }


def compile_revision_guidance(
    reviews: List[Dict],
    discussion: str
) -> str:
    """
    Compile clear, structured guidance for the narrative agent's revision.

    This function creates actionable revision instructions organized by severity,
    ensuring feedback is specific and verifiable.

    Severity levels:
    - BLOCK: Must be addressed - revision will fail otherwise
    - CONCERN: Should be improved - strongly recommended
    - APPROVE: Working well - preserve these elements

    Args:
        reviews: List of agent review dicts with verdict, concern, suggestion, praise
        discussion: Narrative agent's discussion response (may contain revision_plan)

    Returns:
        Compiled guidance string with structured action items
    """
    # Separate feedback by severity
    must_fix = []  # BLOCK-level issues - must be addressed
    should_improve = []  # CONCERN-level issues - strongly recommended
    preserve = []  # PRAISE - keep these elements

    for r in reviews:
        agent = r.get('agent', 'Unknown')
        verdict = r.get('verdict', 'unknown')
        domain = r.get('domain', 'general')

        if verdict == "block" and r.get("suggestion"):
            must_fix.append({
                "agent": agent,
                "domain": domain,
                "concern": r.get("concern", ""),
                "action": r.get("suggestion", "")
            })
        elif verdict == "concern" and r.get("suggestion"):
            should_improve.append({
                "agent": agent,
                "domain": domain,
                "concern": r.get("concern", ""),
                "action": r.get("suggestion", "")
            })

        # Preserve praised elements
        if r.get("praise"):
            preserve.append(f"[{agent}] {r.get('praise')}")

    # Build structured guidance
    guidance_parts = ["=" * 60]
    guidance_parts.append("REVISION GUIDANCE FROM ROUND TABLE")
    guidance_parts.append("=" * 60)

    # MUST FIX section (blocking issues)
    if must_fix:
        guidance_parts.append("\n🚫 MUST FIX (These blocked approval - revision will fail if not addressed):\n")
        for i, item in enumerate(must_fix, 1):
            guidance_parts.append(f"  {i}. [{item['agent']} - {item['domain'].upper()}]")
            guidance_parts.append(f"     ISSUE: {item['concern']}")
            guidance_parts.append(f"     ACTION: {item['action']}")
            guidance_parts.append("")

    # SHOULD IMPROVE section (concerns)
    if should_improve:
        guidance_parts.append("\n⚠️ SHOULD IMPROVE (Concerns raised - address if possible):\n")
        for i, item in enumerate(should_improve, 1):
            guidance_parts.append(f"  {i}. [{item['agent']} - {item['domain'].upper()}]")
            guidance_parts.append(f"     CONCERN: {item['concern']}")
            guidance_parts.append(f"     SUGGESTION: {item['action']}")
            guidance_parts.append("")

    # PRESERVE section (what's working)
    if preserve:
        guidance_parts.append("\n✅ PRESERVE (Do NOT change these - they work well):\n")
        for item in preserve:
            guidance_parts.append(f"  • {item}")
        guidance_parts.append("")

    # Extract narrative agent's revision plan from discussion
    try:
        discussion_data = json.loads(clean_json_output(discussion))
        if "revision_plan" in discussion_data:
            guidance_parts.append("\n📝 YOUR REVISION PLAN (from discussion):\n")
            guidance_parts.append(f"  {discussion_data['revision_plan']}")
            guidance_parts.append("")
    except (json.JSONDecodeError, Exception):
        pass

    # Add verification reminder
    guidance_parts.append("\n" + "=" * 60)
    guidance_parts.append("VERIFICATION CHECKLIST:")
    guidance_parts.append("=" * 60)
    if must_fix:
        guidance_parts.append(f"□ Address ALL {len(must_fix)} blocking issue(s) above")
    if should_improve:
        guidance_parts.append(f"□ Consider {len(should_improve)} improvement(s) above")
    guidance_parts.append("□ Preserve praised elements")
    guidance_parts.append("□ Maintain story's emotional truth")
    guidance_parts.append("□ Keep educational content intact")

    return "\n".join(guidance_parts)


def categorize_reviews(reviews: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Categorize reviews by verdict for easier processing.

    Args:
        reviews: List of review dicts from agents

    Returns:
        Dict with keys 'block', 'concern', 'approve' containing lists of reviews
    """
    categorized = {
        "block": [],
        "concern": [],
        "approve": []
    }

    for review in reviews:
        verdict = review.get("verdict", "approve").lower()
        if verdict in categorized:
            categorized[verdict].append(review)
        else:
            # Unknown verdict defaults to approve
            categorized["approve"].append(review)

    return categorized


def has_blocking_reviews(reviews: List[Dict]) -> bool:
    """
    Check if any reviews have blocking verdicts.

    Args:
        reviews: List of review dicts from agents

    Returns:
        True if any review has verdict='block'
    """
    return any(r.get("verdict", "").lower() == "block" for r in reviews)


def count_by_verdict(reviews: List[Dict]) -> Dict[str, int]:
    """
    Count reviews by verdict type.

    Args:
        reviews: List of review dicts from agents

    Returns:
        Dict with verdict counts: {"block": N, "concern": N, "approve": N}
    """
    counts = {"block": 0, "concern": 0, "approve": 0}

    for review in reviews:
        verdict = review.get("verdict", "approve").lower()
        if verdict in counts:
            counts[verdict] += 1
        else:
            counts["approve"] += 1

    return counts
