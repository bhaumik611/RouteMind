import pytest
from app.core.config import Settings
from app.core.constants import ProviderType
from app.core.exceptions import FreeTierViolationError


def test_default_settings_enforce_free_only():
    """Verify that settings enforce FREE_ONLY by default."""
    cfg = Settings()
    assert cfg.FREE_ONLY is True
    assert cfg.MAX_DAILY_REQUESTS > 0
    assert cfg.MAX_DAILY_TOKENS > 0


def test_openrouter_free_model_validation():
    """Test that openrouter models require :free suffix when FREE_ONLY=True."""
    cfg = Settings(FREE_ONLY=True)

    # Valid free models should pass without exception
    cfg.validate_model_safety(ProviderType.OPENROUTER.value, "meta-llama/llama-3-8b-instruct:free")
    cfg.validate_model_safety(ProviderType.OPENROUTER.value, "mistralai/mistral-7b-instruct:free")

    # Non-free model should raise FreeTierViolationError
    with pytest.raises(FreeTierViolationError) as exc_info:
        cfg.validate_model_safety(ProviderType.OPENROUTER.value, "openai/gpt-4o")

    assert "is not free" in exc_info.value.message
    assert exc_info.value.status_code == 403


def test_gemini_free_model_validation():
    """Test that Gemini models are validated against the free tier whitelist."""
    cfg = Settings(FREE_ONLY=True)

    # Allowed free tier models
    cfg.validate_model_safety(ProviderType.GEMINI.value, "gemini-1.5-flash")
    cfg.validate_model_safety(ProviderType.GEMINI.value, "gemini-1.5-pro")

    # Non-whitelisted/paid model raises FreeTierViolationError
    with pytest.raises(FreeTierViolationError) as exc_info:
        cfg.validate_model_safety(ProviderType.GEMINI.value, "gemini-ultra-paid-version")

    assert exc_info.value.status_code == 403
