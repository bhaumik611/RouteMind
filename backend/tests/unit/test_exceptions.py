from app.core.exceptions import (
    FreeTierViolationError,
    ModelNotFoundError,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitExceededError,
    RouteMindException,
)


def test_exception_serialization():
    exc = FreeTierViolationError(
        message="Model is paid",
        details={"provider": "openrouter", "model": "gpt-4o"},
    )
    serialized = exc.to_dict()

    assert "error" in serialized
    assert serialized["error"]["message"] == "Model is paid"
    assert serialized["error"]["type"] == "FREE_TIER_VIOLATION"
    assert serialized["error"]["status_code"] == 403
    assert serialized["error"]["details"]["model"] == "gpt-4o"


def test_rate_limit_exception_details():
    exc = RateLimitExceededError(message="Tenant rate limit exceeded", retry_after=30)
    data = exc.to_dict()

    assert data["error"]["status_code"] == 429
    assert data["error"]["details"]["retry_after_seconds"] == 30
