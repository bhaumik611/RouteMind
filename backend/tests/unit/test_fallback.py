import pytest
from app.core.constants import ProviderType
from app.providers.local_mock import LocalMockProvider
from app.resiliency.fallback import FallbackEngine
from app.schemas.chat import ChatCompletionRequest, ChatMessage
from app.schemas.provider import ModelMetadata


@pytest.mark.asyncio
async def test_fallback_ordering():
    engine = FallbackEngine()
    primary = ModelMetadata(
        id="gemini/gemini-1.5-flash",
        provider_id=ProviderType.GEMINI,
        model_name="gemini-1.5-flash",
        quality_score=0.86,
    )
    candidate_b = ModelMetadata(
        id="openrouter/meta-llama/llama-3-8b-instruct:free",
        provider_id=ProviderType.OPENROUTER,
        model_name="meta-llama/llama-3-8b-instruct:free",
        quality_score=0.85,
    )
    mock_model = ModelMetadata(
        id="local/mock-free-model",
        provider_id=ProviderType.LOCAL_MOCK,
        model_name="mock-free-model",
        quality_score=0.75,
    )

    ordered = engine.order_fallback_candidates(primary, [primary, candidate_b, mock_model])
    assert ordered[0].id == primary.id
    assert ordered[1].id == candidate_b.id
    assert ordered[2].id == mock_model.id


@pytest.mark.asyncio
async def test_fallback_execution_on_local_mock():
    engine = FallbackEngine()
    mock_model = ModelMetadata(
        id="local/mock-free-model",
        provider_id=ProviderType.LOCAL_MOCK,
        model_name="mock-free-model",
        quality_score=0.75,
    )

    req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")])
    result = await engine.execute_resiliently(req, mock_model, [mock_model])

    assert result.response.content is not None
    assert result.model.model_name == "mock-free-model"
    assert result.is_fallback is False
