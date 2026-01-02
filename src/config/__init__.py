"""Configuration package for Kids Storyteller V2"""

from .settings import Settings, get_settings
from .limits import (
    PROMPT_MAX_LENGTH,
    MESSAGE_MAX_LENGTH,
    CHAPTER_CONTENT_MAX_LENGTH,
    SYNOPSIS_MAX_LENGTH,
    TITLE_MAX_LENGTH,
    TITLE_MIN_LENGTH,
    NAME_MAX_LENGTH,
    NAME_MIN_LENGTH,
)

__all__ = [
    "Settings",
    "get_settings",
    "PROMPT_MAX_LENGTH",
    "MESSAGE_MAX_LENGTH",
    "CHAPTER_CONTENT_MAX_LENGTH",
    "SYNOPSIS_MAX_LENGTH",
    "TITLE_MAX_LENGTH",
    "TITLE_MIN_LENGTH",
    "NAME_MAX_LENGTH",
    "NAME_MIN_LENGTH",
]
