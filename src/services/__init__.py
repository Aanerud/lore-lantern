"""Services package for Lore Lantern"""

from .firebase import FirebaseService
from .llm import LLMService
from .ssml_processor import SSMLProcessor, get_ssml_processor
from .voice import VoiceService, voice_service
from .foundry import FoundryService, get_foundry_service, init_foundry_service
from .llm_router import LLMRouter, get_llm_router, init_llm_router, reset_llm_router
from .adaptive_rate_limiter import (
    AdaptiveRateLimiter,
    get_rate_limiter,
    init_rate_limiter,
    reset_rate_limiter,
    RateLimitError
)
from .text_processing import (
    normalize_reviewer_name,
    normalize_norse_name,
    extract_patronymic,
    extract_first_name
)
from .validation_service import (
    clean_json_output,
    clean_json_for_character,
    auto_fix_character_data
)
from .character_service import (
    normalize_character_names,
    check_relationship_match,
    match_character_identity,
    CharacterService
)
from .review_service import (
    build_review_context,
    compile_revision_guidance,
    categorize_reviews,
    has_blocking_reviews,
    count_by_verdict
)

__all__ = [
    "FirebaseService",
    "LLMService",
    "SSMLProcessor",
    "get_ssml_processor",
    "VoiceService",
    "voice_service",
    "FoundryService",
    "get_foundry_service",
    "init_foundry_service",
    # LLM Router
    "LLMRouter",
    "get_llm_router",
    "init_llm_router",
    "reset_llm_router",
    # Adaptive Rate Limiter
    "AdaptiveRateLimiter",
    "get_rate_limiter",
    "init_rate_limiter",
    "reset_rate_limiter",
    "RateLimitError",
    # Text processing utilities
    "normalize_reviewer_name",
    "normalize_norse_name",
    "extract_patronymic",
    "extract_first_name",
    # Validation utilities
    "clean_json_output",
    "clean_json_for_character",
    "auto_fix_character_data",
    # Character service utilities
    "normalize_character_names",
    "check_relationship_match",
    "match_character_identity",
    "CharacterService",
    # Review service utilities
    "build_review_context",
    "compile_revision_guidance",
    "categorize_reviews",
    "has_blocking_reviews",
    "count_by_verdict",
]
