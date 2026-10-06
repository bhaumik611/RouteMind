from datetime import datetime, timezone
import threading
from typing import Dict, Optional, Tuple
import redis.asyncio as aioredis

from app.core.config import settings
from app.core.exceptions import QuotaExceededError
from app.core.logging import logger


class VirtualQuotaManager:
    """Manages virtual daily token quotas and request limits to prevent free-tier exhaustion."""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self._redis: Optional[aioredis.Redis] = None
        self._memory_tokens: Dict[str, int] = {}
        self._memory_requests: Dict[str, int] = {}
        self._lock = threading.Lock()

    def _get_today_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    async def _get_redis(self) -> Optional[aioredis.Redis]:
        if self._redis is None and self.redis_url:
            try:
                self._redis = aioredis.from_url(
                    self.redis_url, encoding="utf-8", decode_responses=True
                )
                await self._redis.ping()
            except Exception:
                self._redis = None
        return self._redis

    async def check_quota(
        self,
        tenant_id: str,
        estimated_tokens: int = 100,
        daily_token_limit: Optional[int] = None,
        daily_request_limit: Optional[int] = None,
    ) -> None:
        """Checks if tenant has remaining quota for the day."""
        max_tokens = daily_token_limit or settings.MAX_DAILY_TOKENS
        max_reqs = daily_request_limit or settings.MAX_DAILY_REQUESTS
        today = self._get_today_str()

        usage = await self.get_tenant_usage(tenant_id)
        current_tokens = usage["tokens_consumed_today"]
        current_reqs = usage["requests_today"]

        if current_reqs + 1 > max_reqs:
            raise QuotaExceededError(
                message=f"Daily request quota ({max_reqs} requests/day) exceeded for tenant '{tenant_id}'.",
                details={"tenant_id": tenant_id, "limit": max_reqs, "current": current_reqs},
            )

        if current_tokens + estimated_tokens > max_tokens:
            raise QuotaExceededError(
                message=f"Daily virtual token quota ({max_tokens} tokens/day) exceeded for tenant '{tenant_id}'.",
                details={
                    "tenant_id": tenant_id,
                    "limit": max_tokens,
                    "current": current_tokens,
                    "requested": estimated_tokens,
                },
            )

    async def record_usage(self, tenant_id: str, tokens: int) -> None:
        """Increments consumed tokens and request count for today."""
        today = self._get_today_str()
        key_suffix = f"{tenant_id}:{today}"

        client = await self._get_redis()
        if client:
            try:
                pipe = client.pipeline()
                pipe.incrby(f"routemind:quota:{key_suffix}:tokens", max(0, tokens))
                pipe.incr(f"routemind:quota:{key_suffix}:reqs")
                pipe.expire(f"routemind:quota:{key_suffix}:tokens", 172800)  # 48h TTL
                pipe.expire(f"routemind:quota:{key_suffix}:reqs", 172800)
                await pipe.execute()
                return
            except Exception as e:
                logger.warning(f"Redis quota update failed: {str(e)}. Using in-memory tracking.")

        with self._lock:
            self._memory_tokens[key_suffix] = self._memory_tokens.get(key_suffix, 0) + max(0, tokens)
            self._memory_requests[key_suffix] = self._memory_requests.get(key_suffix, 0) + 1

    async def get_tenant_usage(self, tenant_id: str) -> Dict[str, any]:
        """Returns today's usage statistics for a tenant."""
        today = self._get_today_str()
        key_suffix = f"{tenant_id}:{today}"

        client = await self._get_redis()
        if client:
            try:
                pipe = client.pipeline()
                pipe.get(f"routemind:quota:{key_suffix}:tokens")
                pipe.get(f"routemind:quota:{key_suffix}:reqs")
                results = await pipe.execute()

                tokens = int(results[0]) if results[0] else 0
                reqs = int(results[1]) if results[1] else 0
                return {
                    "tenant_id": tenant_id,
                    "date": today,
                    "tokens_consumed_today": tokens,
                    "requests_today": reqs,
                    "max_daily_tokens": settings.MAX_DAILY_TOKENS,
                    "max_daily_requests": settings.MAX_DAILY_REQUESTS,
                    "token_budget_remaining": max(0, settings.MAX_DAILY_TOKENS - tokens),
                }
            except Exception:
                pass

        with self._lock:
            tokens = self._memory_tokens.get(key_suffix, 0)
            reqs = self._memory_requests.get(key_suffix, 0)

        return {
            "tenant_id": tenant_id,
            "date": today,
            "tokens_consumed_today": tokens,
            "requests_today": reqs,
            "max_daily_tokens": settings.MAX_DAILY_TOKENS,
            "max_daily_requests": settings.MAX_DAILY_REQUESTS,
            "token_budget_remaining": max(0, settings.MAX_DAILY_TOKENS - tokens),
        }

    def reset_memory(self):
        """Resets in-memory quota tracking (useful in tests)."""
        with self._lock:
            self._memory_tokens.clear()
            self._memory_requests.clear()


# Global Quota Manager Singleton
quota_manager = VirtualQuotaManager()
