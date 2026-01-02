"""
Adaptive Rate Limiter for LLM API Calls

Features:
1. Per-provider concurrency tracking (Gemini, Claude, Azure, etc.)
2. Dynamic scaling based on 429/rate limit responses
3. Exponential backoff with jitter
4. Async context manager for easy integration
5. Statistics for observability

Usage:
    from src.services.adaptive_rate_limiter import get_rate_limiter

    limiter = get_rate_limiter()

    async with limiter.acquire("azure"):
        response = await some_llm_call()

    # Or with automatic retry:
    result = await limiter.execute_with_retry(
        llm_call_func,
        prompt,
        provider="azure",
        max_retries=3
    )
"""

import asyncio
import time
import random
import logging
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when a rate limit (429) is detected"""

    def __init__(self, provider: str, retry_after: Optional[float] = None):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(f"Rate limit hit for {provider}")


@dataclass
class ProviderState:
    """Tracks state for a single LLM provider"""

    name: str
    max_concurrent: int = 10  # Starting max
    current_concurrent: int = 0
    semaphore: asyncio.Semaphore = field(default=None)

    # Priority lane for dialogue (never blocked by heavy agents)
    priority_semaphore: asyncio.Semaphore = field(default=None)
    priority_max: int = 3  # Always 3 slots reserved for dialogue

    # Metrics
    total_calls: int = 0
    successful_calls: int = 0
    rate_limited_calls: int = 0
    priority_calls: int = 0  # Track dialogue calls
    last_429_time: Optional[float] = None
    consecutive_successes: int = 0

    # Scaling parameters
    min_concurrent: int = 1
    scale_down_factor: float = 0.5  # Reduce by 50% on 429
    scale_up_threshold: int = 20  # Successes before scaling up
    scale_up_increment: int = 2  # Add 2 concurrent slots

    def __post_init__(self):
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.max_concurrent)
        if self.priority_semaphore is None:
            self.priority_semaphore = asyncio.Semaphore(self.priority_max)


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter with per-provider tracking.

    Automatically adjusts concurrency based on API responses:
    - On 429: Reduce max_concurrent by 50%, wait, retry
    - On sustained success: Gradually increase back up
    """

    def __init__(
        self,
        initial_max_concurrent: int = 10,
        min_concurrent: int = 1,
        max_concurrent_ceiling: int = 20,
        base_backoff: float = 2.0,
        max_backoff: float = 60.0,
        jitter: float = 0.5,
    ):
        self.initial_max = initial_max_concurrent
        self.min_concurrent = min_concurrent
        self.max_ceiling = max_concurrent_ceiling
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.jitter = jitter

        self._providers: Dict[str, ProviderState] = {}
        self._lock = asyncio.Lock()

        logger.info(
            f"AdaptiveRateLimiter initialized: max={initial_max_concurrent}, "
            f"min={min_concurrent}, ceiling={max_concurrent_ceiling}"
        )

    def _get_provider_state(self, provider: str) -> ProviderState:
        """Get or create state for a provider"""
        if provider not in self._providers:
            self._providers[provider] = ProviderState(
                name=provider,
                max_concurrent=self.initial_max,
                min_concurrent=self.min_concurrent,
                semaphore=asyncio.Semaphore(self.initial_max),
                priority_semaphore=asyncio.Semaphore(3),  # 3 priority slots for dialogue
            )
            logger.debug(f"Created provider state for '{provider}' with max={self.initial_max}, priority=3")
        return self._providers[provider]

    def _extract_provider(self, model: str) -> str:
        """Extract provider name from model string"""
        if not model:
            return "default"

        model_lower = model.lower()

        if "model-router" in model_lower or model_lower.startswith("azure/"):
            return "azure"
        elif model_lower.startswith("gemini/") or "gemini" in model_lower:
            return "gemini"
        elif "claude" in model_lower or model_lower.startswith("anthropic/"):
            return "anthropic"
        elif model_lower.startswith("openai/") or model_lower.startswith("gpt"):
            return "openai"
        else:
            return "default"

    @asynccontextmanager
    async def acquire(self, provider_or_model: str = "default", priority: str = "normal"):
        """
        Async context manager to acquire a slot for an LLM call.

        Args:
            provider_or_model: Provider name or model string to extract provider from
            priority: "dialogue" for priority lane (never blocked), "normal" for standard lane

        Usage:
            # Standard lane (for heavy agents)
            async with limiter.acquire("azure"):
                response = await llm_call()

            # Priority lane (for dialogue - never blocked by heavy agents)
            async with limiter.acquire("azure", priority="dialogue"):
                response = await dialogue_call()

        Raises:
            RateLimitError: If a rate limit is detected during the call
        """
        provider = self._extract_provider(provider_or_model)
        state = self._get_provider_state(provider)

        # Use priority lane for dialogue - separate semaphore, never blocked by heavy agents
        if priority == "dialogue":
            await state.priority_semaphore.acquire()
            state.priority_calls += 1
            state.total_calls += 1
            try:
                yield
                state.successful_calls += 1
            except Exception as e:
                if self._is_rate_limit_error(e):
                    # Don't scale down for priority calls - they're special
                    logger.warning(f"⚠️ {provider}: Priority call hit rate limit (not scaling down)")
                    raise RateLimitError(provider)
                raise
            finally:
                state.priority_semaphore.release()
        else:
            # Standard lane for heavy agents
            await state.semaphore.acquire()
            state.current_concurrent += 1
            state.total_calls += 1

            try:
                yield
                # Call succeeded
                await self._record_success(provider, state)
            except Exception as e:
                # Check if it's a rate limit error
                if self._is_rate_limit_error(e):
                    await self._handle_rate_limit(provider, state, e)
                    raise RateLimitError(provider)
                raise
            finally:
                state.current_concurrent -= 1
                state.semaphore.release()

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Detect 429/rate limit errors from various providers"""
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

    async def _record_success(self, provider: str, state: ProviderState):
        """Record successful call and potentially scale up"""
        state.successful_calls += 1
        state.consecutive_successes += 1

        # Scale up if we've had enough consecutive successes
        if state.consecutive_successes >= state.scale_up_threshold:
            async with self._lock:
                new_max = min(
                    state.max_concurrent + state.scale_up_increment, self.max_ceiling
                )
                if new_max > state.max_concurrent:
                    old_max = state.max_concurrent
                    state.max_concurrent = new_max
                    # Add more permits to semaphore
                    for _ in range(new_max - old_max):
                        state.semaphore.release()
                    state.consecutive_successes = 0
                    logger.info(
                        f"📈 {provider}: Scaled UP concurrency {old_max} → {new_max} "
                        f"(after {state.scale_up_threshold} successes)"
                    )

    async def _handle_rate_limit(
        self, provider: str, state: ProviderState, error: Exception
    ):
        """Handle rate limit by scaling down and backing off"""
        state.rate_limited_calls += 1
        state.consecutive_successes = 0
        state.last_429_time = time.time()

        async with self._lock:
            # Scale down concurrency
            new_max = max(
                int(state.max_concurrent * state.scale_down_factor),
                state.min_concurrent,
            )

            if new_max < state.max_concurrent:
                old_max = state.max_concurrent
                state.max_concurrent = new_max
                # Note: We can't easily reduce semaphore permits, but the
                # max_concurrent tracking will prevent scaling back up too fast
                logger.warning(
                    f"📉 {provider}: Scaled DOWN concurrency {old_max} → {new_max} "
                    f"(429 rate limit hit)"
                )

        # Calculate backoff with jitter
        retry_after = self._extract_retry_after(error)
        if retry_after:
            backoff = retry_after
        else:
            # Exponential backoff based on number of rate limits
            backoff = min(
                self.base_backoff * (2 ** min(state.rate_limited_calls, 5)),
                self.max_backoff,
            )

        # Add jitter (+-50%)
        jitter_amount = backoff * self.jitter * (random.random() * 2 - 1)
        backoff = max(1.0, backoff + jitter_amount)

        logger.warning(f"⏳ {provider}: Backing off for {backoff:.1f}s")
        await asyncio.sleep(backoff)

    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extract Retry-After header value if present"""
        # Try to extract from exception attributes
        if hasattr(error, "response"):
            response = error.response
            if hasattr(response, "headers"):
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        return float(retry_after)
                    except ValueError:
                        pass
        return None

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        provider: str = "default",
        max_retries: int = 3,
        **kwargs,
    ) -> Any:
        """
        Execute a function with automatic retry on rate limit.

        Usage:
            result = await limiter.execute_with_retry(
                llm_call,
                prompt,
                provider="azure",
                max_retries=3
            )
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                async with self.acquire(provider):
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return await asyncio.to_thread(func, *args, **kwargs)
            except RateLimitError as e:
                last_error = e
                logger.warning(
                    f"🔄 {provider}: Retry {attempt + 1}/{max_retries} after rate limit"
                )
                if attempt == max_retries - 1:
                    raise
            except Exception:
                # Non-rate-limit error, don't retry
                raise

        raise last_error

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all providers"""
        return {
            provider: {
                "max_concurrent": state.max_concurrent,
                "current_concurrent": state.current_concurrent,
                "priority_slots": state.priority_max,
                "total_calls": state.total_calls,
                "successful_calls": state.successful_calls,
                "priority_calls": state.priority_calls,  # Dialogue calls
                "rate_limited_calls": state.rate_limited_calls,
                "success_rate": (
                    round(state.successful_calls / state.total_calls * 100, 1)
                    if state.total_calls > 0
                    else 100.0
                ),
                "last_429": state.last_429_time,
                "consecutive_successes": state.consecutive_successes,
            }
            for provider, state in self._providers.items()
        }

    def get_provider_concurrency(self, provider: str) -> int:
        """Get current max concurrency for a provider"""
        if provider in self._providers:
            return self._providers[provider].max_concurrent
        return self.initial_max

    def log_stats(self):
        """Log current statistics for all providers"""
        stats = self.get_stats()
        if not stats:
            logger.info("📊 AdaptiveRateLimiter: No providers tracked yet")
            return

        logger.info("📊 AdaptiveRateLimiter Statistics:")
        for provider, data in stats.items():
            logger.info(
                f"   {provider}: {data['successful_calls']}/{data['total_calls']} calls "
                f"({data['success_rate']}% success), "
                f"max_concurrent={data['max_concurrent']}, "
                f"rate_limits={data['rate_limited_calls']}"
            )


# Singleton instance
_rate_limiter: Optional[AdaptiveRateLimiter] = None


def get_rate_limiter() -> AdaptiveRateLimiter:
    """Get the singleton rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = AdaptiveRateLimiter()
    return _rate_limiter


def init_rate_limiter(
    initial_max_concurrent: int = 10, **kwargs
) -> AdaptiveRateLimiter:
    """Initialize the rate limiter with custom settings"""
    global _rate_limiter
    _rate_limiter = AdaptiveRateLimiter(
        initial_max_concurrent=initial_max_concurrent, **kwargs
    )
    return _rate_limiter


def reset_rate_limiter():
    """Reset the singleton (for testing)"""
    global _rate_limiter
    _rate_limiter = None
