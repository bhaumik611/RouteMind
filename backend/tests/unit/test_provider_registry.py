import pytest
from app.core.constants import ProviderType
from app.core.exceptions import ModelNotFoundError, ProviderUnavailableError
from app.providers.local_mock import LocalMockProvider
from app.providers.registry import ProviderRegistry


def test_provider_registry_lifecycle():
    registry = ProviderRegistry()
    mock_provider = LocalMockProvider()

    # Register
    registry.register(mock_provider)
    assert registry.get(ProviderType.LOCAL_MOCK) == mock_provider

    # List models
    models = registry.list_models(free_only=True)
    assert len(models) >= 3
    assert any(m.model_name == "mock-free-model" for m in models)

    # Get model by name & ID
    model_by_name = registry.get_model("mock-free-model")
    assert model_by_name.is_free is True

    model_by_id = registry.get_model("local_mock/mock-free-model")
    assert model_by_id.model_name == "mock-free-model"


def test_model_not_found():
    registry = ProviderRegistry()
    with pytest.raises(ModelNotFoundError):
        registry.get_model("non-existent-model-xyz")


def test_provider_not_found():
    registry = ProviderRegistry()
    with pytest.raises(ProviderUnavailableError):
        registry.get(ProviderType.GEMINI)
