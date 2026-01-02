"""
Configuration management for Lore Lantern

Loads environment variables and provides application settings.
"""

from pydantic_settings import BaseSettings
from typing import Optional, Dict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    app_name: str = "Lore Lantern"
    port: int = 3000
    log_level: str = "INFO"

    # CORS Configuration
    # Default "*" allows all origins (suitable for development and public APIs)
    # For production with specific frontends, set to comma-separated list:
    #   CORS_ALLOWED_ORIGINS=https://lorelantern.com,https://lorelantern.no
    cors_allowed_origins: str = "*"

    # OpenAI (optional, if using)
    openai_api_key: Optional[str] = None

    # Claude (Anthropic)
    claude_api_key: Optional[str] = None

    # Google Gemini
    google_genai_api_key: str

    # Firebase Configuration
    firebase_api_key: str
    firebase_auth_domain: str
    firebase_database_url: str
    firebase_project_id: str
    firebase_storage_bucket: str
    firebase_messaging_sender_id: str
    firebase_app_id: str
    google_application_credentials: Optional[str] = None

    # Firebase Service Account (optional - for direct credential usage)
    # DEPRECATED: Firebase is being replaced by Azure SQL + Blob Storage
    firebase_client_email: Optional[str] = None
    firebase_private_key: Optional[str] = None
    firebase_private_key_id: Optional[str] = None

    # =========================================================================
    # Azure SQL Database Configuration
    # Replaces Firebase Realtime Database for structured data
    # =========================================================================
    azure_sql_server: Optional[str] = None  # e.g., lorelantern-db.database.windows.net
    azure_sql_database: Optional[str] = None  # e.g., lorelantern
    azure_sql_username: Optional[str] = None
    azure_sql_password: Optional[str] = None

    # =========================================================================
    # Azure Blob Storage Configuration
    # For audio file storage (MP3 chapter audio)
    # =========================================================================
    azure_blob_connection_string: Optional[str] = None
    azure_blob_container: str = "lorelantern-audio"

    # =========================================================================
    # Storage Backend Selection
    # Set USE_AZURE_SQL=true to use Azure SQL instead of Firebase
    # =========================================================================
    use_azure_sql: bool = False  # Feature flag for gradual migration

    # =========================================================================
    # Microsoft Foundry Configuration (Unified AI Platform)
    # Model Router auto-routes prompts to optimal model (GPT, Claude, etc.)
    # https://ai.azure.com - Azure's unified AI platform
    # =========================================================================
    foundry_endpoint: Optional[str] = None  # e.g., https://foundry-lorelantern.cognitiveservices.azure.com
    foundry_api_key: Optional[str] = None
    foundry_api_version: str = "2024-12-01-preview"  # Model Router API version (official docs)
    use_foundry: bool = False  # Feature flag - when True, use Foundry instead of direct APIs

    # Model Router routing mode: "quality", "cost", "balanced"
    # - quality: Routes to best-performing model for complex tasks
    # - cost: Optimizes for cost while staying within 5-6% of best quality
    # - balanced: Default, balances cost and quality (within 1-2% of best)
    foundry_routing_mode: str = "quality"  # For story generation, prioritize quality

    # =========================================================================
    # Azure AI Foundry - Claude Sonnet 4.5 (Dedicated Deployment)
    # Direct access to Claude via Azure (alternative to Anthropic API)
    # Uses Anthropic-compatible API format but hosted on Azure
    # =========================================================================
    azure_claude_endpoint: Optional[str] = None  # e.g., https://xxx.services.ai.azure.com/anthropic/v1/messages
    azure_claude_api_key: Optional[str] = None
    azure_claude_model: str = "claude-sonnet-4-5"

    # =========================================================================
    # Agent Model Configuration
    # =========================================================================
    # MOVED TO: src/config/models.yaml
    #
    # All agent model configuration is now centralized in models.yaml and
    # managed by the LLMRouter service (src/services/llm_router.py).
    #
    # A/B Testing via environment variables:
    #   TEST_ROUNDTABLE_MODEL=gemini-3-flash-preview python tests/e2e_test.py
    #   TEST_NARRATIVE_MODEL=claude-sonnet-4-5 python tests/e2e_test.py
    #   TEST_STRUCTURE_MODEL=gpt-5 python tests/e2e_test.py
    #
    # See models.yaml for per-agent and group configuration.
    # =========================================================================

    # LLM Configuration (legacy - kept for backward compatibility)
    llm_model: str = "gemini-1.5-flash"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048

    # Agent Configuration
    max_task_retries: int = 3
    task_timeout_seconds: int = 30

    # Voice Direction Configuration (ElevenLabs Audio Tags)
    # The VoiceDirectorAgent transforms prose into expressive narration
    # with ElevenLabs audio tags like [whispers], [laughing], [excited]
    # This adds emotional depth that differentiates from plain TTS
    voice_direction_enabled: bool = True  # Enabled for ElevenLabs audio tags

    # CompanionAgent Configuration
    # The CompanionAgent is always available and never blocked by CrewAI
    # Uses Anthropic models exclusively (to avoid Gemini quota issues)
    companion_fast_model: str = "claude-3-haiku-20240307"        # Fastest responses (~500ms-1s)
    companion_quality_model: str = "claude-sonnet-4-5-20250929"  # Rich character spotlights
    companion_proactive_interval: int = 30                      # Seconds between teasers
    companion_max_teasers: int = 4                              # Max teasers before Chapter 1

    # =========================================================================
    # Streaming Dialogue Configuration
    # When enabled, dialogue text streams to client as tokens arrive from LLM
    # Audio is generated in background and arrives after text completes
    # This provides immediate feedback and never blocks on heavy agent work
    # =========================================================================
    # DISABLED: litellm doesn't properly handle Azure Model Router streaming format
    # Re-enable when FoundryService has streaming support (see plan file)
    enable_streaming_dialogue: bool = False  # Disabled - use non-streaming path

    # One-Chapter-At-A-Time Configuration
    # Instead of generating multiple chapters upfront, we generate Chapter 1 on init,
    # then prefetch the next chapter when the current one starts playing.
    # This provides faster initial story ready time and on-demand generation.
    initial_chapters_to_write: int = 1  # Only Chapter 1 on story initialization
    enable_chapter_prefetch: bool = True  # Prefetch next chapter when current starts playing

    # Voice Interface (optional for Phase 1)
    enable_voice: bool = False

    # Development Mode - disables TTS to save costs
    dev_mode: bool = True  # Set to False in production
    disable_tts: bool = True  # Disable TTS audio generation during development

    # ElevenLabs TTS Configuration (Primary Provider)
    # Uses Eleven v3 - most expressive model with audio tags support
    # Supports [emotion] tags like [whispers], [laughing], [excited]
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_model: str = "eleven_v3"  # Most expressive, supports audio tags
    elevenlabs_output_format: str = "mp3_44100_128"  # Good quality MP3

    # ElevenLabs Voice IDs by language
    # Custom voices selected for each language for proper pronunciation
    elevenlabs_voices: Dict[str, str] = {
        "en": "7p1Ofvcwsv7UBPoFNcpI",  # Custom English narrator
        "no": "xF681s0UeE04gsf0mVsJ",  # Norwegian narrator (no Danish/Swedish mixing)
        "es": "21m00Tcm4TlvDq8ikWAM",  # Rachel (v3 handles Spanish)
    }

    # TTS Provider Configuration
    # USE_ELEVENLABS=true enables ElevenLabs for premium narration
    # Otherwise: Dialogue→OpenAI, Narration→Speechify
    use_elevenlabs: bool = False  # Set USE_ELEVENLABS=true to enable ElevenLabs for narration

    # Speechify TTS Configuration (Story Narration)
    # Used for chapter audio when ElevenLabs is not enabled
    # Norwegian is in beta (nb-NO)
    speechify_api_key: Optional[str] = None

    # OpenAI TTS Configuration (Companion Dialogue)
    # Uses gpt-4o-mini-tts for steerable, expressive dialogue
    openai_tts_model: str = "gpt-4o-mini-tts"  # Supports instructions for voice style
    openai_tts_voice: str = "nova"  # Warm, friendly voice for children

    # =========================================================================
    # Direct Audio Output (LLM + TTS in Single Call)
    # Uses GPT-4o-mini-audio-preview to generate text AND audio in one API call
    # This reduces dialogue latency by ~50% (single call vs LLM→TTS pipeline)
    # =========================================================================
    use_direct_audio: bool = True  # Enable single-call LLM+audio for dialogue
    direct_audio_model: str = "gpt-4o-mini-audio-preview"  # Faster than gpt-4o-audio-preview
    direct_audio_voice: str = "nova"  # Voice for audio output

    # Debug Configuration (ENABLED for troubleshooting)
    debug_agent_io: bool = True  # Log agent inputs/outputs
    debug_storage: bool = True   # Log storage operations (Azure SQL or Firebase)
    debug_api_calls: bool = True  # Log LLM API call details
    debug_log_dir: str = "logs/debug"  # Directory for debug logs
    debug_log_format: str = "text"  # "json" or "text" - using text for easier reading

    # =========================================================================
    # Ollama Configuration (Local Language Models)
    # Used for language-specific text refinement (e.g., Norwegian with Borealis)
    # =========================================================================
    ollama_base_url: Optional[str] = None  # e.g., http://192.168.0.157:11434

    class Config:
        env_file = ".env"
        case_sensitive = False

    def get_firebase_credentials_dict(self) -> Optional[Dict]:
        """
        Get Firebase credentials as a dict from environment variables.

        Returns None if credentials are not available.

        .. deprecated::
            Firebase is being replaced by Azure SQL + Blob Storage.
            Set USE_AZURE_SQL=true to use the new storage backend.
        """
        import warnings
        if self.firebase_client_email and self.firebase_private_key:
            warnings.warn(
                "Firebase credentials are deprecated. Migrate to Azure SQL + Blob Storage "
                "by setting USE_AZURE_SQL=true. See docs/ARCHITECTURE_DECISIONS.md for details.",
                DeprecationWarning,
                stacklevel=2
            )
            return {
                "type": "service_account",
                "project_id": self.firebase_project_id,
                "private_key_id": self.firebase_private_key_id or "",
                "private_key": self.firebase_private_key.replace("\\n", "\n"),  # Handle escaped newlines
                "client_email": self.firebase_client_email,
                "client_id": "",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            }
        return None


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Using lru_cache ensures we only load settings once.
    """
    return Settings()
