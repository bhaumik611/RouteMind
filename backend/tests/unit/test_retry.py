import pytest
from app.core.exceptions import FreeTierViolationError, ProviderUnavailableError
from app.resiliency.retry import RetryPolicy


def test_compute_delay():
    policy = RetryPolicy(max_retries=3, initial_delay_seconds=0.1, backoff_factor=2.0, jitter=False)
    assert policy.compute_delay(0) == 0.1
    assert policy.compute_delay(1) == 0.2
    assert policy.compute_delay(2) == 0.4


def test_non_retryable_safety_violation():
    policy = RetryPolicy()
    exc = FreeTierViolationError("Paid model blocked")
    assert policy.is_retryable(exc) is False


@pytest.mark.asyncio
async def test_retry_success_after_failure():
    policy = RetryPolicy(max_retries=2, initial_delay_seconds=0.01, jitter=False)
    attempts = 0

    async def flaky_call():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ProviderUnavailableError("Temporary network hiccup")
        return "Success"

    result = await policy.execute_with_retry(flaky_call, provider_name="test")
    assert result == "Success"
    assert attempts == 2


@pytest.mark.asyncio
async def test_retry_exhaustion():
    policy = RetryPolicy(max_retries=2, initial_delay_seconds=0.01, jitter=False)
    attempts = 0

    async def always_failing_call():
        nonlocal attempts
        attempts += 1
        raise ProviderUnavailableError("Permanent down")

    with pytest.raises(ProviderUnavailableError):
        await policy.execute_with_retry(always_failing_call, provider_name="test")

    assert attempts == 3  # Initial + 2 retries
