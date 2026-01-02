"""
Writers' Room Discussion Prompt

Nnedi Okofor responds to concerns from the Round Table reviewers,
facilitating the writers' room discussion.
"""


def get_discussion_prompt(
    chapter_number: int,
    concerns_summary: str,
    suggestions_summary: str,
    chapter_content_preview: str
) -> str:
    """
    Generate the writers' room discussion prompt for Nnedi.

    Let Nnedi respond to concerns, facilitating the discussion.

    Args:
        chapter_number: Chapter being discussed
        concerns_summary: Formatted concerns from reviewers
        suggestions_summary: Formatted suggestions from reviewers
        chapter_content_preview: First ~2000 chars of chapter for context

    Returns:
        Formatted prompt string for the discussion
    """
    return f"""WRITERS' ROOM DISCUSSION - Chapter {chapter_number}

You are Nnedi Okofor, the Africanfuturist writer. Your colleagues have raised these concerns:

CONCERNS:
{concerns_summary}

SUGGESTIONS:
{suggestions_summary}

YOUR CHAPTER (for reference):
{chapter_content_preview}...

As the writer, respond to each concern:
- Do you agree or disagree?
- How would you address it in revision?
- Is there context they might be missing?

Then propose a revision approach that addresses the valid concerns
while preserving what works.

Output JSON:
{{
    "responses": [
        {{"to": "Guillermo", "response": "..."}},
        {{"to": "Bill", "response": "..."}},
        {{"to": "Clarissa", "response": "..."}}
    ],
    "revision_plan": "Here's how I'll revise...",
    "preserved_elements": ["What I'll keep..."]
}}"""
