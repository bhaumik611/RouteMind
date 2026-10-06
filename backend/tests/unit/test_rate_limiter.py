import pytest
from app.core.exceptions import RateLimitExceededError
from app.services.rate_limiter import SlidingWindowRateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_allows_under_limit():
    limiter = SlidingWindowRateLimiter(redis_url="")
    key = "test-tenant-allow"

    for i in range(5):
        allowed, remaining, _ = await limiter.check_rate_limit(key, max_requests=10, window_seconds=60)
        assert allowed is True
        assert remaining == 10 - (i + 1)


@pytest.mark.asyncio
async def test_rate_limiter_blocks_over_limit():
    limiter = SlidingWindowRateLimiter(redis_url="")
    key = "test-tenant-block"

    # Fill quota of 3 requests
    for _ in range(3):
        allowed, _, _ = await limiter.check_rate_limit(key, max_requests=3, window_seconds=60)
        assert allowed is True

    # 4th request must be denied
    allowed, remaining, retry_after = await limiter.check_rate_limit(key, max_requests=3, window_seconds=60)
    assert allowed is False
    assert remaining == 0
    assert retry_after > 0


@pytest.mark.asyncio
async def test_rate_limiter_enforce_raises_exception():
    limiter = SlidingWindowRateLimiter(redis_url="")
    key = "test-tenant-enforce"

    for _ in range(2):
        await limiter.enforce(key, max_requests=2, window_seconds=60)

    with pytest.raises(RateLimitExceededError) as exc:
        await limiter.enforce(key, max_requests=2, window_seconds=60)

    assert exc.value.status_code == 429
