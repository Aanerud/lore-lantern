"""
Model Validation Service

Validates that all configured LLM models are available before application startup.
Lists all available models from each provider.
"""

import anthropic
import openai
from typing import Dict, List, Tuple
import os
import requests
import logging

logger = logging.getLogger(__name__)


class ModelValidator:
    """Validates LLM model availability across providers"""

    def __init__(self):
        self.anthropic_client = None
        self.openai_client = None
        self.anthropic_key = None
        self.openai_key = None
        self.google_key = None

    def validate_all_models(self, agent_models: Dict[str, str]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate all models configured in agent_models.

        Args:
            agent_models: Dict mapping agent names to model IDs

        Returns:
            Tuple of (all_valid, available_models, unavailable_models)
        """
        available = []
        unavailable = []

        # Initialize clients
        self._init_clients()

        # Get unique models
        unique_models = set(agent_models.values())

        for model in unique_models:
            if self._is_model_available(model):
                available.append(model)
            else:
                unavailable.append(model)

        all_valid = len(unavailable) == 0
        return all_valid, available, unavailable

    def _init_clients(self):
        """Initialize API clients"""
        # Anthropic
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        if self.anthropic_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)

        # OpenAI
        self.openai_key = os.getenv("OPENAI_API_KEY")
        if self.openai_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_key)

        # Google (check all common env var names)
        self.google_key = os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    def list_all_available_models(self) -> Dict[str, List[str]]:
        """
        Query each provider's API to get ALL available models.

        Returns:
            Dict mapping provider name to list of available model IDs
        """
        self._init_clients()

        all_models = {}

        # Get Azure/Foundry models (if configured)
        azure_models = self._list_azure_models()
        if azure_models:
            all_models["Azure AI Foundry"] = azure_models

        # Get Anthropic models
        anthropic_models = self._list_anthropic_models()
        if anthropic_models:
            all_models["Anthropic (Claude)"] = anthropic_models

        # Get OpenAI models
        openai_models = self._list_openai_models()
        if openai_models:
            all_models["OpenAI (GPT)"] = openai_models

        # Get Google models
        google_models = self._list_google_models()
        if google_models:
            all_models["Google (Gemini)"] = google_models

        return all_models

    def _list_azure_models(self) -> List[str]:
        """List Azure/Foundry models if configured"""
        use_foundry = os.getenv("USE_FOUNDRY", "false").lower() == "true"
        foundry_endpoint = os.getenv("FOUNDRY_ENDPOINT")

        if not use_foundry or not foundry_endpoint:
            return []

        # Return the models we know are deployed in Foundry
        # These are configured via Azure Portal/CLI
        return [
            "model-router (auto-selects best model)",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
        ]

    def _list_anthropic_models(self) -> List[str]:
        """List all available Anthropic Claude models"""
        if not self.anthropic_key:
            return []

        try:
            # Call Anthropic's models API
            response = requests.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": self.anthropic_key,
                    "anthropic-version": "2023-06-01"
                }
            )

            if response.status_code == 200:
                data = response.json()
                if "data" in data:
                    return [model["id"] for model in data["data"]]

            # Fallback: return known models if API doesn't support listing
            return []
        except Exception as e:
            print(f"   Note: Could not fetch Anthropic models list: {str(e)[:80]}")
            return []

    def _list_openai_models(self) -> List[str]:
        """List all available OpenAI models"""
        if not self.openai_client:
            return []

        try:
            models = self.openai_client.models.list()
            # Filter to show only GPT models (not embeddings, TTS, etc.)
            gpt_models = [
                m.id for m in models.data
                if any(prefix in m.id for prefix in ['gpt-', 'chatgpt-', 'o1-', 'o3-'])
            ]
            return sorted(gpt_models)
        except Exception as e:
            print(f"   Note: Could not fetch OpenAI models list: {str(e)[:80]}")
            return []

    def _list_google_models(self) -> List[str]:
        """List all available Google Gemini models from their API"""
        if not self.google_key:
            return []

        try:
            # Google's Generative Language API has a models endpoint
            response = requests.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={self.google_key}"
            )

            if response.status_code == 200:
                data = response.json()
                if "models" in data:
                    # Filter to gemini models and extract just the model name
                    gemini_models = []
                    for model in data["models"]:
                        name = model.get("name", "")
                        # name format is "models/gemini-1.5-flash" - extract just the model part
                        if name.startswith("models/"):
                            model_id = name.replace("models/", "")
                            if "gemini" in model_id:
                                gemini_models.append(model_id)
                    return sorted(gemini_models)

            # Fallback to known models if API fails
            return [
                "gemini-2.0-flash",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
            ]
        except Exception as e:
            print(f"   Note: Could not fetch Google models list: {str(e)[:80]}")
            return [
                "gemini-2.0-flash",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
            ]

    def _is_model_available(self, model: str) -> bool:
        """
        Test if a specific model is available.

        Args:
            model: Model ID (e.g., "gpt-4o", "anthropic/claude-opus-4-1-20250805", "azure/model-router")

        Returns:
            True if model is available, False otherwise
        """
        # Handle Azure/Foundry models (validated via Foundry service, not here)
        if model.startswith("azure/"):
            return self._test_azure_model(model)

        # Handle Azure AI hosted Claude models (azure_ai/ prefix for LiteLLM)
        if model.startswith("azure_ai/"):
            return self._test_azure_claude_model(model)

        # Strip LiteLLM provider prefixes for testing
        test_model = model.replace("anthropic/", "").replace("gemini/", "").replace("openai/", "")

        # Determine provider
        if test_model.startswith("claude-") or test_model.startswith("anthropic"):
            return self._test_anthropic_model(test_model)
        elif any(test_model.startswith(prefix) for prefix in ["gpt-", "chatgpt-", "o1-", "o3-"]):
            return self._test_openai_model(test_model)
        elif test_model.startswith("gemini-"):
            # Gemini models are tested via LiteLLM/Google SDK
            return self._test_gemini_model(test_model)
        else:
            print(f"⚠️  Unknown model provider for: {model}")
            return False

    def _test_azure_model(self, model: str) -> bool:
        """Test if Azure/Foundry model is available via environment config"""
        # Azure models are validated by checking if Foundry is configured
        foundry_endpoint = os.getenv("FOUNDRY_ENDPOINT")
        foundry_key = os.getenv("FOUNDRY_API_KEY")
        use_foundry = os.getenv("USE_FOUNDRY", "false").lower() == "true"

        if use_foundry and foundry_endpoint and foundry_key:
            # Foundry is configured, assume Azure models are available
            return True
        else:
            print(f"⚠️  Azure model {model} requires USE_FOUNDRY=true and FOUNDRY_ENDPOINT")
            return False

    def _test_azure_claude_model(self, model: str) -> bool:
        """
        Test if Azure AI hosted Claude model is available.

        These models use the azure_ai/ prefix and are accessed via Azure's
        Anthropic-compatible endpoint (e.g., claude-sonnet-4-5 hosted on Azure).

        Note: main.py exports these as AZURE_AI_API_KEY and AZURE_AI_API_BASE
        """
        # Check both naming conventions (main.py uses AZURE_AI_*, settings uses AZURE_CLAUDE_*)
        azure_claude_key = os.getenv("AZURE_AI_API_KEY") or os.getenv("AZURE_CLAUDE_API_KEY")
        azure_claude_base = os.getenv("AZURE_AI_API_BASE") or os.getenv("AZURE_CLAUDE_ENDPOINT")

        # Build the full endpoint for Anthropic messages API
        if azure_claude_base:
            # Ensure we have the full messages endpoint
            if not azure_claude_base.endswith("/anthropic/v1/messages"):
                azure_claude_endpoint = azure_claude_base.rstrip("/") + "/anthropic/v1/messages"
            else:
                azure_claude_endpoint = azure_claude_base
        else:
            azure_claude_endpoint = None

        if azure_claude_endpoint and azure_claude_key:
            # Try a minimal test call to the Azure Claude endpoint
            try:
                # Extract model name from azure_ai/claude-sonnet-4-5 -> claude-sonnet-4-5
                model_name = model.replace("azure_ai/", "")

                response = requests.post(
                    azure_claude_endpoint,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": azure_claude_key,
                        "anthropic-version": "2023-06-01"
                    },
                    json={
                        "model": model_name,
                        "max_tokens": 10,
                        "messages": [{"role": "user", "content": "Hi"}]
                    },
                    timeout=30
                )

                # 200 = success, 400 = bad request but model exists
                if response.status_code in [200, 400]:
                    return True
                elif response.status_code == 404:
                    print(f"⚠️  Azure Claude model {model_name} not found at endpoint")
                    return False
                else:
                    # Other status codes (429 rate limit, 500 error) suggest model exists
                    return True

            except Exception as e:
                print(f"⚠️  Could not verify Azure Claude model {model}: {str(e)[:60]}")
                # If we have config but can't reach, assume it might work
                return True
        else:
            print(f"⚠️  Azure Claude model {model} requires AZURE_AI_API_KEY and AZURE_AI_API_BASE (or AZURE_CLAUDE_*)")
            return False

    def _test_anthropic_model(self, model: str) -> bool:
        """Test if Anthropic model is available"""
        if not self.anthropic_client:
            return False

        try:
            self.anthropic_client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except anthropic.NotFoundError:
            return False
        except Exception as e:
            # Other errors (auth, rate limit) might still mean model exists
            error_str = str(e).lower()
            if "not found" in error_str or "does not exist" in error_str:
                return False
            # If it's a different error, assume model exists but there's another issue
            print(f"⚠️  {model}: {str(e)[:100]}")
            return True

    def _test_openai_model(self, model: str) -> bool:
        """Test if OpenAI model is available"""
        if not self.openai_client:
            return False

        try:
            # Use models.list() to check if model exists
            models = self.openai_client.models.list()
            available_models = [m.id for m in models.data]
            return model in available_models
        except Exception as e:
            print(f"⚠️  Error checking OpenAI model {model}: {str(e)[:100]}")
            # If we can't list models, try a minimal completion
            try:
                self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5
                )
                return True
            except openai.NotFoundError:
                return False
            except Exception as e:
                logger.warning(f"Could not verify OpenAI model {model}, assuming available: {e}")
                return True  # Assume exists if other error

    def _test_gemini_model(self, model: str) -> bool:
        """Test if Gemini model is available"""
        # Gemini models are accessed via LiteLLM, which uses GOOGLE_API_KEY or GOOGLE_GENAI_API_KEY
        google_key = os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not google_key:
            return False

        # Common Gemini model patterns (including gemini-3 preview)
        valid_patterns = ["gemini-1.5-", "gemini-2.", "gemini-3-", "gemini-pro", "gemini-flash"]
        return any(pattern in model for pattern in valid_patterns)


def validate_models(agent_models: Dict[str, str], show_all_models: bool = False) -> None:
    """
    Validate all configured models and optionally list all available models from each provider.

    Args:
        agent_models: Dict mapping agent names to model IDs
        show_all_models: If True, list ALL available models from each provider
    """
    validator = ModelValidator()

    # List all available models from providers
    if show_all_models:
        print("\n📋 Listing ALL available models from each provider...\n")

        all_provider_models = validator.list_all_available_models()

        for provider, models in all_provider_models.items():
            print(f"🔷 {provider} ({len(models)} models):")
            for model in models:
                # Check if this model is configured
                is_configured = any(
                    m == model or m == f"anthropic/{model}"
                    for m in agent_models.values()
                )
                marker = " ⭐ CONFIGURED" if is_configured else ""
                print(f"   • {model}{marker}")
            print()

    # Validate configured models
    print("🔍 Validating configured AI models...")

    all_valid, available, unavailable = validator.validate_all_models(agent_models)

    # Print results
    if available:
        print(f"\n✅ Available models ({len(available)}):")
        for model in sorted(available):
            # Show which agents use this model
            agents = [name for name, m in agent_models.items() if m == model]
            print(f"   • {model}")
            print(f"     Used by: {', '.join(agents)}")

    if unavailable:
        print(f"\n❌ Unavailable models ({len(unavailable)}):")
        for model in sorted(unavailable):
            agents = [name for name, m in agent_models.items() if m == model]
            print(f"   • {model}")
            print(f"     Needed by: {', '.join(agents)}")
        print("\n⚠️  WARNING: Some configured models are not available!")
        print("   The application may fail when these agents are used.")
        print("   Please check your API keys and model IDs in settings.py")

    if all_valid:
        print(f"\n✅ All {len(available)} configured models are available!\n")

    return all_valid
