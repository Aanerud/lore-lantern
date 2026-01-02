"""
Centralized Validation Limits

All content length limits in one place for consistency.
Import these in both API routes and Pydantic models.
"""

# =============================================================================
# USER INPUT LIMITS
# =============================================================================

# Story prompts (including reference material pasted by users)
PROMPT_MAX_LENGTH = 20000

# General message length (chat messages, questions)
MESSAGE_MAX_LENGTH = 5000

# =============================================================================
# CONTENT LIMITS
# =============================================================================

# Chapter content (narration text)
CHAPTER_CONTENT_MAX_LENGTH = 100000

# Synopsis/summary text
SYNOPSIS_MAX_LENGTH = 10000

# Title fields
TITLE_MAX_LENGTH = 100
TITLE_MIN_LENGTH = 3

# =============================================================================
# FIELD-SPECIFIC LIMITS
# =============================================================================

# Character/skill names
NAME_MAX_LENGTH = 100
NAME_MIN_LENGTH = 2

# Factual statements (for fact-checking)
STATEMENT_MAX_LENGTH = 500
STATEMENT_MIN_LENGTH = 3

# Skill names (D&D style)
SKILL_NAME_MAX_LENGTH = 50
SKILL_NAME_MIN_LENGTH = 2

# =============================================================================
# COLLECTION LIMITS
# =============================================================================

# Maximum chapters per story (soft limit, can be exceeded)
MAX_CHAPTERS_DEFAULT = 20

# Maximum characters per story
MAX_CHARACTERS_DEFAULT = 50

# Maximum dialogue history entries to keep
MAX_DIALOGUE_HISTORY = 100
