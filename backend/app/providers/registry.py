from typing import Dict, List, Optional

from app.core.config import settings
from app.core.constants import ProviderType
from app.core.exceptions import ConfigurationError, FreeTierViolationError, ModelNotFoundError, ProviderUnavailableError
from app.core.logging import logger
from app.providers.base import BaseLLMProvider
from app.providers.gemini import GeminiProvider
from app.providers.local_mock import LocalMockProvider
from app.providers.openrouter import OpenRouterProvider
from app.schemas.provider import ModelMetadata, ProviderMetadata


class ProviderRegistry:
    """Centralized Registry for LLM Providers and Free-tier Catalog Validation."""

    def __init__(self):
        self._providers: Dict[ProviderType, BaseLLMProvider] = {}
        self._models_cache: Dict[str, ModelMetadata] = {}

    def register(self, provider: BaseLLMProvider) -> None:
        """Registers a provider and updates the model lookup index."""
        self._providers[provider.provider_type] = provider
        logger.info(f"Registered provider: {provider.name} [{provider.provider_type.value}]")
        self._refresh_model_cache()

    def unregister(self, provider_type: ProviderType) -> None:
        if provider_type in self._providers:
            del self._providers[provider_type]
            self._refresh_model_cache()

    def get(self, provider_type: ProviderType) -> BaseLLMProvider:
        """Retrieves a registered provider adapter."""
        if provider_type not in self._providers:
            raise ProviderUnavailableError(f"Provider '{provider_type.value}' is not registered.")
        return self._providers[provider_type]

    def _refresh_model_cache(self) -> None:
        """Rebuilds the fast model lookup dictionary."""
        self._models_cache.clear()
        for provider in self._providers.values():
            for model in provider.get_available_models():
                # Enforce FREE_ONLY verification during registry indexing
                if settings.FREE_ONLY and not model.is_free:
                    logger.warning(
                        f"Skipping paid model '{model.id}' because FREE_ONLY=True",
                        extra={"extra_data": {"model_id": model.id}},
                    )
                    continue
                self._models_cache[model.id] = model
                # Also index by plain model_name
                self._models_cache[model.model_name] = model

    def list_providers(self) -> List[ProviderMetadata]:
        """Returns structured metadata for all registered providers."""
        return [provider.get_metadata() for provider in self._providers.values()]

    def list_models(self, free_only: bool = True) -> List[ModelMetadata]:
        """Returns all registered models, strictly filtered if free_only=True."""
        models = list(self._models_cache.values())
        seen_ids = set()
        deduped = []
        for m in models:
            if m.id not in seen_ids:
                seen_ids.add(m.id)
                if free_only and not m.is_free:
                    continue
                deduped.append(m)
        return deduped

    def get_model(self, model_id: str) -> ModelMetadata:
        """Retrieves model metadata by full ID or model name."""
        if model_id in self._models_cache:
            return self._models_cache[model_id]
        raise ModelNotFoundError(model_id)


def create_default_registry() -> ProviderRegistry:
    """Factory creating a ProviderRegistry loaded with OpenRouter, Gemini, and Local Mock providers."""
    registry = ProviderRegistry()
    registry.register(LocalMockProvider())
    registry.register(OpenRouterProvider())
    registry.register(GeminiProvider())
    return registry


# Global Provider Registry Singleton
registry = create_default_registry()
