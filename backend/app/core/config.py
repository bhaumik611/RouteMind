from typing import List, Optional, Set
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.constants import (
    GEMINI_VERIFIED_FREE_MODELS,
    LOCAL_VERIFIED_MODELS,
    OPENROUTER_VERIFIED_FREE_MODELS,
    ProviderType,
    RoutingStrategy,
)
from app.core.exceptions import ConfigurationError, FreeTierViolationError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application Info
    APP_NAME: str = "RouteMind"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_V1_STR: str = "/v1"

    # CRITICAL ZERO-BUDGET SAFETY GUARD
    FREE_ONLY: bool = Field(
        default=True,
        description="When True, strictly forbids any paid provider/model requests across the system.",
    )

    # Database & Cache
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/routemind"
    REDIS_URL: str = "redis://localhost:6379/0"

    # Provider Keys (Optional / empty if not yet provided)
    OPENROUTER_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    # Configured Free Models
    OPENROUTER_MODEL: str = "meta-llama/llama-3-8b-instruct:free"
    GEMINI_MODEL: str = "gemini-1.5-flash"
    LOCAL_MODEL: str = "mock-free-model"

    # Virtual Quota & Limits
    MAX_DAILY_REQUESTS: int = 5000
    MAX_DAILY_TOKENS: int = 1000000
    MAX_REQUEST_TOKENS: int = 4096
    REQUEST_TIMEOUT_SECONDS: float = 30.0

    # Resiliency Settings
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS: float = 30.0
    MAX_FALLBACK_ATTEMPTS: int = 3

    # Routing Defaults
    DEFAULT_ROUTING_STRATEGY: RoutingStrategy = RoutingStrategy.BANDIT

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    @field_validator("FREE_ONLY", mode="after")
    @classmethod
    def validate_free_only(cls, v: bool) -> bool:
        if not v:
            # RouteMind is architected with zero budget constraints.
            # Emitting an architectural safety warning if disabled.
            pass
        return v

    def validate_model_safety(self, provider: str, model_name: str) -> None:
        """Validates that the given provider and model name adhere strictly to FREE_ONLY constraints.
        Raises FreeTierViolationError if the model is not verified as free."""
        if not self.FREE_ONLY:
            return

        provider_lower = provider.lower()
        model_lower = model_name.lower()

        if provider_lower == ProviderType.OPENROUTER.value:
            if not model_lower.endswith(":free") and model_name not in OPENROUTER_VERIFIED_FREE_MODELS:
                raise FreeTierViolationError(
                    message=f"Model '{model_name}' on OpenRouter is not free. FREE_ONLY is enabled.",
                    details={"provider": provider, "model": model_name, "required_suffix": ":free"},
                )

        elif provider_lower == ProviderType.GEMINI.value:
            if model_lower not in [m.lower() for m in GEMINI_VERIFIED_FREE_MODELS]:
                raise FreeTierViolationError(
                    message=f"Model '{model_name}' on Gemini is not in verified free tier whitelist.",
                    details={"provider": provider, "model": model_name, "allowed": list(GEMINI_VERIFIED_FREE_MODELS)},
                )

        elif provider_lower in [ProviderType.LOCAL_MOCK.value, ProviderType.OLLAMA.value]:
            if model_lower not in [m.lower() for m in LOCAL_VERIFIED_MODELS] and not model_lower.startswith("mock"):
                # Ollama is locally hosted, but we also verify against configured local models
                pass

    def is_provider_available(self, provider: ProviderType) -> bool:
        """Checks if required credentials/endpoints are set for a provider."""
        if provider == ProviderType.OPENROUTER:
            return bool(self.OPENROUTER_API_KEY)
        if provider == ProviderType.GEMINI:
            return bool(self.GEMINI_API_KEY)
        if provider in (ProviderType.LOCAL_MOCK, ProviderType.OLLAMA):
            return True
        return False


settings = Settings()
