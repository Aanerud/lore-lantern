"""
Lore Lantern - Main Application

Production-quality AI-powered storytelling system for children.
Multi-language voice-enabled storytelling lantern device.
https://lorelantern.com
"""

# SQLite fix for Azure App Service - must be at the very top before any other imports
# ChromaDB requires sqlite3 >= 3.35.0, but Azure's Linux image has an older version
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # pysqlite3 not installed, use system sqlite3

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import sys
import os
import tempfile
from datetime import datetime

from src.config import get_settings
from src.services import FirebaseService
from src.services.database import DatabaseService
from src.services.blob_storage import BlobStorageService
from src.services.logger import init_logger
from src.crew import StoryCrewCoordinator
from src.api.routes import router, set_coordinator, set_blob_storage_service, setup_prefetch_listener
from src.api.websocket import router as websocket_router, set_coordinator as set_ws_coordinator
from src.agents.companion import CompanionAgent, set_companion_agent
from src.services.events import story_events
import asyncio

# Configure logging to both file and console
# Use /tmp for Azure App Service (read-only filesystem) or local logs for development
if os.environ.get('WEBSITE_SITE_NAME'):  # Running on Azure App Service
    log_dir = Path(tempfile.gettempdir()) / "lorelantern_logs"
else:
    log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"lorelantern_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Create formatters
file_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_formatter = logging.Formatter('%(message)s')

# File handler (detailed logs)
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(file_formatter)

# Console handler (user-friendly output)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_formatter)

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger(__name__)
logger.info(f"📝 Logging to: {log_file}")


# Global services
firebase_service: FirebaseService = None
database_service: DatabaseService = None
blob_storage_service: BlobStorageService = None
coordinator: StoryCrewCoordinator = None
companion_agent: CompanionAgent = None
storage_service = None  # Will be either firebase_service or database_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle management for the application.

    Initializes services on startup, cleans up on shutdown.
    """
    # Startup
    global firebase_service, database_service, blob_storage_service, storage_service, coordinator

    settings = get_settings()

    print("🏮 Initializing Lore Lantern...")

    # Set event loop for non-blocking event emission from worker threads
    story_events.set_event_loop(asyncio.get_running_loop())

    # Initialize logger with settings
    app_logger = init_logger(settings=settings)

    # Show debug status
    debug_flags = []
    if settings.debug_agent_io:
        debug_flags.append("Agent I/O")
    if settings.debug_storage:
        debug_flags.append("Storage")
    if settings.debug_api_calls:
        debug_flags.append("API Calls")

    if debug_flags:
        print(f"🐛 Debug logging enabled: {', '.join(debug_flags)}")
        print(f"📊 Debug logs: {settings.debug_log_dir}/")

    # Configure Ollama for local language models (Norwegian refinement via Borealis)
    if settings.ollama_base_url:
        os.environ["OLLAMA_API_BASE"] = settings.ollama_base_url
        print(f"🦙 Ollama configured: {settings.ollama_base_url}")

    # Initialize Storage Service (Azure SQL or Firebase)
    if settings.use_azure_sql:
        print("📊 Connecting to Azure SQL Database...")
        database_service = DatabaseService(
            server=settings.azure_sql_server,
            database=settings.azure_sql_database,
            username=settings.azure_sql_username,
            password=settings.azure_sql_password,
            logger=app_logger
        )
        database_service.initialize()
        storage_service = database_service
        print("✅ Azure SQL connected")

        # Initialize Blob Storage for audio files
        if settings.azure_blob_connection_string:
            print("📦 Connecting to Azure Blob Storage...")
            blob_storage_service = BlobStorageService(
                connection_string=settings.azure_blob_connection_string,
                container_name=settings.azure_blob_container,
                logger=app_logger
            )
            blob_storage_service.initialize()
            set_blob_storage_service(blob_storage_service)  # Make available to routes
            print("✅ Azure Blob Storage connected")
    else:
        # Legacy Firebase initialization
        print("📊 Connecting to Firebase...")

        # Try to get credentials from environment variables first, then fall back to file
        creds_dict = settings.get_firebase_credentials_dict()

        firebase_service = FirebaseService(
            database_url=settings.firebase_database_url,
            credentials_dict=creds_dict,
            credentials_path=settings.google_application_credentials,
            logger=app_logger
        )
        firebase_service.initialize()
        storage_service = firebase_service
        print("✅ Firebase connected")

    # Set environment variables for LiteLLM/CrewAI
    # (os is imported globally at top of file)

    # Configure for Microsoft Foundry if enabled
    if settings.use_foundry and settings.foundry_endpoint:
        print("🔵 Configuring LiteLLM for Microsoft Foundry Model Router...")
        os.environ["AZURE_API_KEY"] = settings.foundry_api_key or ""
        os.environ["AZURE_API_BASE"] = settings.foundry_endpoint
        os.environ["AZURE_API_VERSION"] = settings.foundry_api_version
        print(f"   Foundry endpoint: {settings.foundry_endpoint[:40]}...")

        # Initialize global Foundry service singleton for shared access (e.g., Structure V2)
        from src.services.foundry import init_foundry_service
        init_foundry_service(
            endpoint=settings.foundry_endpoint,
            api_key=settings.foundry_api_key,
            api_version=settings.foundry_api_version,
            routing_mode=getattr(settings, 'foundry_routing_mode', 'balanced')
        )
        print("   ✅ Foundry service singleton initialized")

    # Initialize LLM Router (centralized model configuration)
    # This loads models.yaml and applies any TEST_*_MODEL env var overrides
    from src.services.llm_router import init_llm_router
    llm_router = init_llm_router()
    print("📋 LLM Router initialized")
    llm_router.log_configuration()

    # Always set Google API key - needed for model validation and potential non-Foundry agents
    if settings.google_genai_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_genai_api_key
        os.environ["GEMINI_API_KEY"] = settings.google_genai_api_key

    # Always set Anthropic API key if available (needed for direct Claude model calls
    # even when Foundry is enabled, e.g., NarrativeAgent uses claude-sonnet-4-5)
    if settings.claude_api_key:
        os.environ["ANTHROPIC_API_KEY"] = settings.claude_api_key
        masked_claude = f"{settings.claude_api_key[:12]}...{settings.claude_api_key[-4:]}" if len(settings.claude_api_key) > 16 else "***"
        print(f"✅ ANTHROPIC_API_KEY: {masked_claude}")

    # Configure Azure AI (Claude Sonnet 4.5 via Azure AI Foundry)
    # This is a dedicated Claude deployment for quality-critical tasks like NarrativeAgent
    if settings.azure_claude_endpoint and settings.azure_claude_api_key:
        # Extract base URL (remove /anthropic/v1/messages suffix for LiteLLM)
        azure_claude_base = settings.azure_claude_endpoint.replace("/anthropic/v1/messages", "")
        os.environ["AZURE_AI_API_KEY"] = settings.azure_claude_api_key
        os.environ["AZURE_AI_API_BASE"] = azure_claude_base
        masked_key = f"{settings.azure_claude_api_key[:12]}...{settings.azure_claude_api_key[-4:]}" if len(settings.azure_claude_api_key) > 16 else "***"
        print(f"✅ AZURE_AI_API_KEY (Claude): {masked_key}")
        print(f"   Azure Claude endpoint: {azure_claude_base[:50]}...")

    # OpenAI is still needed for TTS (direct audio)
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key

    # Validate critical environment variables
    print("🔍 Validating environment variables...")
    if not settings.openai_api_key:
        print("❌ OPENAI_API_KEY is missing! Voice features will not work.")
        print("   💡 Set OPENAI_API_KEY in .env file")
    else:
        masked_key = f"{settings.openai_api_key[:8]}...{settings.openai_api_key[-4:]}" if len(settings.openai_api_key) > 12 else "***"
        print(f"✅ OPENAI_API_KEY: {masked_key}")

    if not settings.google_genai_api_key:
        print("⚠️  GOOGLE_API_KEY is missing! AI agents may not work.")
    else:
        masked_key = f"{settings.google_genai_api_key[:8]}...{settings.google_genai_api_key[-4:]}" if len(settings.google_genai_api_key) > 12 else "***"
        print(f"✅ GOOGLE_API_KEY: {masked_key}")

    # Initialize model tracker to capture actual model used by LiteLLM
    # This allows us to show which model (e.g., "gpt-oss-120b", "DeepSeek-V3.1")
    # was actually selected by model-router in observatory events
    from src.services.model_tracker import init_model_tracker
    init_model_tracker()
    print("📊 Model tracker initialized (LiteLLM callback registered)")

    # Initialize adaptive rate limiter for parallel LLM calls
    # Starts at 10 concurrent, scales down on 429s, scales up after 20 successes
    from src.services.adaptive_rate_limiter import init_rate_limiter
    rate_limiter = init_rate_limiter(
        initial_max_concurrent=10,
        min_concurrent=1,
        max_concurrent_ceiling=20
    )
    print("🚦 Adaptive rate limiter initialized (max 10 concurrent → auto-scaling)")

    # Validate AI models using LLM Router configuration
    # A/B testing overrides are logged by the LLM Router during initialization
    from src.services.model_validator import validate_models
    from src.services.llm_router import get_llm_router
    llm_router = get_llm_router()
    selected_models = {
        agent: llm_router.get_model_for_agent(agent)
        for agent in ["structure", "character", "narrative", "factcheck",
                      "line_editor", "continuity", "tension", "dialogue", "voice_director"]
    }
    validate_models(selected_models, show_all_models=True)

    # Initialize Coordinator (creates all agents)
    # Agents self-configure from LLM Router (models.yaml)
    print("👥 Creating agent crew...")
    coordinator = StoryCrewCoordinator(storage_service, logger=app_logger)
    set_coordinator(coordinator)
    set_ws_coordinator(coordinator)  # Also set for WebSocket handler
    print("✅ Agent crew assembled")

    # Test voice service initialization
    print("🧪 Testing voice service...")
    from src.services.voice import voice_service
    # Don't pass voice_name - let service use its default for the active provider
    test_audio = await voice_service.text_to_speech("Testing voice service.")
    if test_audio:
        print(f"✅ Voice service test PASSED: Generated {len(test_audio)} bytes")
    else:
        print("❌ Voice service test FAILED: No audio generated")
        print("   ⚠️  Voice features may not work properly!")

    # Initialize CompanionAgent (always-available front-face)
    print("🤝 Initializing CompanionAgent...")
    companion_agent = CompanionAgent(
        firebase_service=storage_service,  # Now accepts either Firebase or Database service
        foundry_endpoint=settings.foundry_endpoint,
        foundry_api_key=settings.foundry_api_key,
        use_foundry=settings.use_foundry,
        proactive_interval=settings.companion_proactive_interval,
        max_teasers=settings.companion_max_teasers
    )
    set_companion_agent(companion_agent)
    print(f"✅ CompanionAgent ready (Hanan persona, Foundry={settings.use_foundry})")

    # Setup one-chapter-at-a-time prefetch listener
    setup_prefetch_listener()
    print("📚 One-chapter-at-a-time prefetch enabled")

    print(f"🏮 Lore Lantern ready on port {settings.port}!")
    print(f"🎤 Voice Interface: http://localhost:{settings.port}/")
    print(f"📚 API Documentation: http://localhost:{settings.port}/docs")

    yield

    # Shutdown
    print("👋 Shutting down Lore Lantern...")
    if firebase_service:
        firebase_service.shutdown()


# Create FastAPI app
app = FastAPI(
    title="Lore Lantern",
    description="""
    Multi-language voice-enabled AI storytelling for children.

    A magical lantern device that tells personalized stories in any language.
    https://lorelantern.com

    Features:
    - Multi-agent story creation (DialogueAgent, StructureAgent, CharacterAgent, NarrativeAgent, FactCheckAgent)
    - Educational content integration
    - Multi-language voice interface (speaks in child's native language)
    - Real-time story generation
    - Parent progress tracking

    Built with CrewAI, FastAPI, and Claude Sonnet 4.5.
    """,
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
# Configure allowed origins from environment variable (default: "*" for all origins)
# For production with specific frontends: CORS_ALLOWED_ORIGINS=https://lorelantern.com,https://lorelantern.no
_settings = get_settings()
_cors_origins = (
    ["*"] if _settings.cors_allowed_origins == "*"
    else [origin.strip() for origin in _settings.cors_allowed_origins.split(",")]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validation error handler - log details for debugging
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    import sys
    errors = exc.errors()
    # Build detailed error message for Azure logging
    error_details = []
    for error in errors:
        input_val = error.get('input', 'N/A')
        # Truncate long inputs for readability
        if isinstance(input_val, str) and len(input_val) > 100:
            input_val = input_val[:100] + "..."
        detail = f"{error['loc']}: {error['msg']} (input: {input_val})"
        error_details.append(detail)

    full_msg = f"❌ Validation Error on {request.url.path}: " + " | ".join(error_details)
    logger.error(full_msg)

    return JSONResponse(
        status_code=422,
        content={"detail": errors}
    )

# Global exception handler - catch unhandled exceptions to prevent crashes
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch all unhandled exceptions to prevent server crashes.
    Logs the error and returns a friendly error message.
    """
    import traceback
    error_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    # Log full traceback
    logger.error(f"❌ UNHANDLED EXCEPTION [{error_id}]")
    logger.error(f"   Path: {request.url.path}")
    logger.error(f"   Method: {request.method}")
    logger.error(f"   Error: {type(exc).__name__}: {exc}")
    logger.error(f"   Traceback:\n{traceback.format_exc()}")

    # Return friendly error to client
    # NOTE: Never expose exception details to clients, even in debug mode
    # Use error_id to look up details in server logs
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_id": error_id,
            "message": "An unexpected error occurred. Please try again."
        }
    )

# Include routes
app.include_router(router)
app.include_router(websocket_router)

# Mount static files for voice interface
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    print(f"📁 Static files mounted from: {static_path}")


@app.get("/")
async def root():
    """Root endpoint - serves voice interface"""
    index_path = static_path / "index.html"
    if index_path.exists():
        with open(index_path) as f:
            return HTMLResponse(content=f.read())

    # Fallback to API info
    return {
        "message": "Welcome to Lore Lantern!",
        "voice_interface": "/static/index.html",
        "docs": "/docs",
        "health": "/api/health",
        "version": "2.0.0"
    }


def main():
    """Run the application"""
    settings = get_settings()

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=False,  # Disabled - WatchFiles causing hangs. Restart manually after code changes.
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
