"""
Microsoft Foundry Service - Unified AI Platform Client

Provides a unified client for Microsoft Foundry's Model Router, which automatically
routes prompts to the optimal model (GPT, Claude, DeepSeek, Llama, Grok) based on
the selected routing mode (quality, cost, balanced).

Architecture:
- Model Router deployment handles all LLM requests
- Single endpoint for all AI operations
- Supports both API key and Managed Identity authentication

Usage:
    from src.services.foundry import FoundryService

    foundry = FoundryService(
        endpoint="https://foundry-lorelantern.cognitiveservices.azure.com",
        api_key="your-api-key"  # Or None for Managed Identity
    )

    response = await foundry.chat_completion(
        messages=[{"role": "user", "content": "Hello!"}],
        routing_mode="quality"
    )
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any, Union
from openai import AzureOpenAI, AsyncAzureOpenAI
from src.services.adaptive_rate_limiter import RateLimitError

logger = logging.getLogger(__name__)


def _is_rate_limit_error(error: Exception) -> bool:
    """Detect if an exception is a rate limit (429) error from Azure/OpenAI."""
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    # Check error type names
    if "ratelimit" in error_type or "toomanyrequests" in error_type:
        return True

    # Check error message content
    rate_limit_indicators = [
        "429",
        "rate limit",
        "rate_limit",
        "ratelimit",
        "too many requests",
        "quota exceeded",
        "requests per minute",
        "tokens per minute",
        "tpm limit",
        "rpm limit",
        "resource exhausted",
    ]

    return any(indicator in error_str for indicator in rate_limit_indicators)


class FoundryService:
    """
    Unified client for Microsoft Foundry Model Router.

    The Model Router automatically routes prompts to the optimal underlying model
    (GPT-4.1, Claude Sonnet 4.5, etc.) based on prompt complexity and routing mode.

    Attributes:
        endpoint: Azure Cognitive Services endpoint
        api_version: API version for Model Router (default: 2024-12-01-preview)
        routing_mode: Default routing mode (quality/cost/balanced)
    """

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        api_version: str = "2024-12-01-preview",
        routing_mode: str = "quality"
    ):
        """
        Initialize the Foundry client.

        Args:
            endpoint: Azure AI Foundry endpoint URL
            api_key: API key for authentication (None to use Managed Identity)
            api_version: API version for Model Router
            routing_mode: Default routing mode ("quality", "cost", "balanced")
        """
        self.endpoint = endpoint
        self.api_version = api_version
        self.default_routing_mode = routing_mode

        # Track last model actually used (for observatory/debugging)
        self._last_model_used = None

        # Initialize clients (sync and async)
        if api_key:
            logger.info(f"Initializing Foundry with API key authentication")
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )
            self.async_client = AsyncAzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )
        else:
            # Use Managed Identity
            logger.info(f"Initializing Foundry with Managed Identity")
            try:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider

                credential = DefaultAzureCredential()
                token_provider = get_bearer_token_provider(
                    credential,
                    "https://cognitiveservices.azure.com/.default"
                )

                self.client = AzureOpenAI(
                    azure_endpoint=endpoint,
                    azure_ad_token_provider=token_provider,
                    api_version=api_version,
                )
                self.async_client = AsyncAzureOpenAI(
                    azure_endpoint=endpoint,
                    azure_ad_token_provider=token_provider,
                    api_version=api_version,
                )
            except ImportError:
                raise ImportError(
                    "azure-identity package required for Managed Identity. "
                    "Install with: pip install azure-identity"
                )

        logger.info(f"Foundry client initialized: endpoint={endpoint}, routing_mode={routing_mode}")

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        routing_mode: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to Azure OpenAI via Foundry.

        Supports both Model Router (when available) and direct deployment access.

        Args:
            messages: List of message dicts with "role" and "content"
            routing_mode: Routing mode for Model Router (quality/cost/balanced)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            model: Optional deployment name override (default: gpt-4o-mini for balanced, gpt-4o for quality)
            reasoning_effort: For reasoning models (gpt-5-mini, o1, etc.) - controls thinking depth
                              Values: "low", "medium", "high" (default varies by model)
                              Lower = faster response, fewer reasoning tokens
            **kwargs: Additional parameters passed to the API

        Returns:
            Dict with response content and metadata

        Example:
            response = await foundry.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Tell me a story about Vikings."}
                ],
                routing_mode="quality",
                reasoning_effort="low"  # Fast response for simple queries
            )
            print(response["content"])  # The response text
            print(response["model"])    # Which model was used
        """
        mode = routing_mode or self.default_routing_mode

        # Use Model Router - it auto-selects the best model (GPT, Claude, etc.)
        deployment = model or "model-router"

        try:
            # Log context being sent (truncated for readability)
            if messages:
                system_msg = next((m['content'] for m in messages if m.get('role') == 'system'), None)
                user_msg = next((m['content'] for m in messages if m.get('role') == 'user'), None)

                effort_str = f", reasoning={reasoning_effort}" if reasoning_effort else ""
                logger.info(f"🔷 Foundry Request: deployment={deployment}, temp={temperature}, max_tokens={max_tokens}{effort_str}")
                if system_msg:
                    # Extract agent persona from system message
                    persona = system_msg[:100].replace('\n', ' ')
                    logger.info(f"   📝 System: {persona}...")
                if user_msg:
                    # Log first 150 chars of user context
                    context_preview = user_msg[:150].replace('\n', ' ')
                    logger.info(f"   💬 Context: {context_preview}...")

            # Build API call parameters
            api_params = {
                "model": deployment,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }

            # Add reasoning_effort for reasoning models (gpt-5-mini, o1, etc.)
            # This reduces "thinking tokens" and speeds up responses
            if reasoning_effort:
                api_params["reasoning_effort"] = reasoning_effort

            # Model Router auto-selects the optimal model based on prompt complexity
            response = await self.async_client.chat.completions.create(**api_params)

            # Extract response content
            result = {
                "content": response.choices[0].message.content,
                "model": response.model,  # Which underlying model was used
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "routing_mode": mode,
                "finish_reason": response.choices[0].finish_reason,
            }

            # Track last model used (for observatory/debugging access)
            self._last_model_used = response.model

            # Extract reasoning tokens if available (for reasoning models like gpt-5-nano)
            reasoning_tokens = 0
            if hasattr(response.usage, 'completion_tokens_details'):
                details = response.usage.completion_tokens_details
                if hasattr(details, 'reasoning_tokens'):
                    reasoning_tokens = details.reasoning_tokens

            # Log model selection prominently
            logger.info(
                f"🤖 Foundry Model Selected: {result['model']} | "
                f"tokens: {result['usage']['total_tokens']} (prompt: {result['usage']['prompt_tokens']}, "
                f"completion: {result['usage']['completion_tokens']}"
                f"{f', reasoning: {reasoning_tokens}' if reasoning_tokens else ''})"
            )

            return result

        except Exception as e:
            logger.error(f"Foundry chat completion failed: {e}")
            # Detect rate limit errors and re-raise as RateLimitError for adaptive handling
            if _is_rate_limit_error(e):
                raise RateLimitError("azure") from e
            raise

    def chat_completion_sync(
        self,
        messages: List[Dict[str, str]],
        routing_mode: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synchronous version of chat_completion.

        Use this for non-async contexts (e.g., CrewAI tasks).
        """
        mode = routing_mode or self.default_routing_mode

        # Use Model Router - it auto-selects the best model (GPT, Claude, etc.)
        deployment = model or "model-router"

        try:
            # Log context being sent (truncated for readability)
            if messages:
                system_msg = next((m['content'] for m in messages if m.get('role') == 'system'), None)
                user_msg = next((m['content'] for m in messages if m.get('role') == 'user'), None)

                logger.info(f"🔷 Foundry Request (sync): deployment={deployment}, temp={temperature}, max_tokens={max_tokens}")
                if system_msg:
                    persona = system_msg[:100].replace('\n', ' ')
                    logger.info(f"   📝 System: {persona}...")
                if user_msg:
                    context_preview = user_msg[:150].replace('\n', ' ')
                    logger.info(f"   💬 Context: {context_preview}...")

            # Model Router auto-selects the optimal model based on prompt complexity
            response = self.client.chat.completions.create(
                model=deployment,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            result = {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "routing_mode": mode,
                "finish_reason": response.choices[0].finish_reason,
            }

            # Track last model used (for observatory/debugging access)
            self._last_model_used = response.model

            # Extract reasoning tokens if available
            reasoning_tokens = 0
            if hasattr(response.usage, 'completion_tokens_details'):
                details = response.usage.completion_tokens_details
                if hasattr(details, 'reasoning_tokens'):
                    reasoning_tokens = details.reasoning_tokens

            logger.info(
                f"🤖 Foundry Model Selected (sync): {result['model']} | "
                f"tokens: {result['usage']['total_tokens']} (prompt: {result['usage']['prompt_tokens']}, "
                f"completion: {result['usage']['completion_tokens']}"
                f"{f', reasoning: {reasoning_tokens}' if reasoning_tokens else ''})"
            )

            return result

        except Exception as e:
            logger.error(f"Foundry chat completion (sync) failed: {e}")
            # Detect rate limit errors and re-raise as RateLimitError for adaptive handling
            if _is_rate_limit_error(e):
                raise RateLimitError("azure") from e
            raise

    async def classify_input(
        self,
        text: str,
        classification_prompt: str,
        routing_mode: str = "balanced"
    ) -> str:
        """
        Classify user input using Model Router.

        Uses "balanced" mode by default for fast, cost-effective classification.

        Args:
            text: User input text to classify
            classification_prompt: System prompt with classification instructions
            routing_mode: Routing mode (default: balanced for speed)

        Returns:
            Classification result as string
        """
        messages = [
            {"role": "system", "content": classification_prompt},
            {"role": "user", "content": text}
        ]

        response = await self.chat_completion(
            messages=messages,
            routing_mode=routing_mode,
            max_tokens=100,  # Classifications are short
            temperature=0.1  # Low temperature for consistent classification
        )

        return response["content"]


# Singleton instance (initialized lazily)
_foundry_service: Optional[FoundryService] = None


def get_foundry_service() -> Optional[FoundryService]:
    """Get the global Foundry service instance."""
    return _foundry_service


def init_foundry_service(
    endpoint: str,
    api_key: Optional[str] = None,
    api_version: str = "2024-12-01-preview",
    routing_mode: str = "quality"
) -> FoundryService:
    """
    Initialize the global Foundry service.

    Call this at application startup when use_foundry=True.
    """
    global _foundry_service
    _foundry_service = FoundryService(
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        routing_mode=routing_mode
    )
    return _foundry_service
