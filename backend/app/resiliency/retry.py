import asyncio
import random
from typing import Callable, Optional, Tuple, Type

from app.core.exceptions import FreeTierViolationError, RouteMindException
from app.core.logging import logger


class RetryPolicy:
    """Configurable Exponential Backoff with Jitter for transient provider errors."""

    def __init__(
        self,
        max_retries: int = 2,
        initial_delay_seconds: float = 0.1,
        backoff_factor: float = 2.0,
        max_delay_seconds: float = 2.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay_seconds
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay_seconds
        self.jitter = jitter

    def compute_delay(self, attempt: int) -> float:
        """Computes exponential delay with optional full jitter."""
        delay = self.initial_delay * (self.backoff_factor ** attempt)
        delay = min(delay, self.max_delay)
        if self.jitter:
            delay = random.uniform(0.0, delay)
        return round(delay, 4)

    def is_retryable(self, exception: Exception) -> bool:
        """Determines if an error is transient (e.g. network timeout/429/5xx) vs non-retryable."""
        # Never retry client validation or safety violations
        if isinstance(exception, FreeTierViolationError):
            return False
        if isinstance(exception, RouteMindException) and exception.status_code in (400, 401, 403, 404, 422):
            return False
        return True

    async def execute_with_retry(
        self,
        coro_func: Callable[[], any],
        provider_name: str = "unknown",
    ):
        """Executes a coroutine function with configured retry policy."""
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                return await coro_func()
            except Exception as e:
                last_exception = e
                if attempt == self.max_retries or not self.is_retryable(e):
                    raise e

                delay = self.compute_delay(attempt)
                logger.warning(
                    f"Retry {attempt + 1}/{self.max_retries} for [{provider_name}] after {delay}s due to: {str(e)}",
                    extra={"extra_data": {"attempt": attempt + 1, "provider": provider_name, "delay_s": delay}},
                )
                await asyncio.sleep(delay)

        if last_exception:
            raise last_exception
