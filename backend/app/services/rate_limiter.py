import time
from typing import Dict, Optional, Tuple
import threading
import redis.asyncio as aioredis

from app.core.config import settings
from app.core.exceptions import RateLimitExceededError
from app.core.logging import logger


class SlidingWindowRateLimiter:
    """Sliding Window Rate Limiter with Redis backend and in-memory fallback."""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self._redis: Optional[aioredis.Redis] = None
        self._memory_cache: Dict[str, list] = {}
        self._lock = threading.Lock()

    async def _get_redis(self) -> Optional[aioredis.Redis]:
        if self._redis is None and self.redis_url:
            try:
                self._redis = aioredis.from_url(
                    self.redis_url, encoding="utf-8", decode_responses=True
                )
                # Quick ping with short timeout
                await self._redis.ping()
            except Exception as e:
                logger.warning(
                    f"Redis unavailable for rate limiting ({str(e)}). Using thread-safe in-memory rate limiter.",
                    extra={"extra_data": {"redis_url": self.redis_url}},
                )
                self._redis = None
        return self._redis

    async def check_rate_limit(
        self,
        key: str,
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> Tuple[bool, int, int]:
        """Checks if key is within rate limit.
        Returns: (is_allowed: bool, remaining_requests: int, retry_after_seconds: int)
        """
        now = time.time()
        window_start = now - window_seconds

        client = await self._get_redis()
        if client:
            try:
                redis_key = f"routemind:rl:{key}"
                pipe = client.pipeline()
                # Remove requests older than sliding window
                pipe.zremrangebyscore(redis_key, 0, window_start)
                # Count current requests in window
                pipe.zcard(redis_key)
                # Add current request timestamp
                pipe.zadd(redis_key, {str(now): now})
                # Set TTL
                pipe.expire(redis_key, window_seconds + 5)
                results = await pipe.execute()

                current_count = results[1]
                if current_count >= max_requests:
                    # Remove the request that just exceeded limit
                    await client.zrem(redis_key, str(now))
                    retry_after = int(window_seconds - (now - window_start))
                    return False, 0, max(1, retry_after)

                remaining = max(0, max_requests - current_count - 1)
                return True, remaining, 0
            except Exception as e:
                logger.warning(f"Redis rate limit check failed: {str(e)}. Falling back to in-memory.")

        # In-Memory Fallback
        with self._lock:
            timestamps = self._memory_cache.get(key, [])
            # Filter timestamps within window
            timestamps = [ts for ts in timestamps if ts > window_start]

            if len(timestamps) >= max_requests:
                oldest_ts = timestamps[0]
                retry_after = int(window_seconds - (now - oldest_ts))
                self._memory_cache[key] = timestamps
                return False, 0, max(1, retry_after)

            timestamps.append(now)
            self._memory_cache[key] = timestamps
            remaining = max(0, max_requests - len(timestamps))
            return True, remaining, 0

    async def enforce(
        self,
        identifier: str,
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> None:
        """Enforces rate limit, raising RateLimitExceededError if violated."""
        allowed, remaining, retry_after = await self.check_rate_limit(
            identifier, max_requests, window_seconds
        )
        if not allowed:
            raise RateLimitExceededError(
                message=f"Rate limit of {max_requests} requests per {window_seconds}s exceeded for '{identifier}'.",
                retry_after=retry_after,
                details={"identifier": identifier, "limit": max_requests, "window_seconds": window_seconds},
            )


# Global Rate Limiter Singleton
rate_limiter = SlidingWindowRateLimiter()
