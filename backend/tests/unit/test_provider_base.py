import pytest
from app.core.constants import ProviderType
from app.core.exceptions import FreeTierViolationError
from app.providers.base import BaseLLMProvider
from app.providers.local_mock import LocalMockProvider
from app.schemas.chat import ChatCompletionRequest, ChatMessage
from app.schemas.provider import ModelMetadata


class MockOpenRouterProvider(LocalMockProvider):
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OPENROUTER

    @property
    def name(self) -> str:
        return "Mock OpenRouter"


@pytest.mark.asyncio
async def test_mock_provider_completion():
    provider = LocalMockProvider()
    request = ChatCompletionRequest(
        model="mock-free-model",
        messages=[ChatMessage(role="user", content="Test message")],
    )

    response = await provider.complete(request)
    assert response.provider.value == "local_mock"
    assert response.model == "mock-free-model"
    assert "Successfully processed message 'Test message" in response.content
    assert response.latency_ms > 0
    assert response.usage.total_tokens > 0

    await provider.close()


@pytest.mark.asyncio
async def test_mock_provider_streaming():
    provider = LocalMockProvider()
    request = ChatCompletionRequest(
        model="mock-free-model",
        messages=[ChatMessage(role="user", content="Stream test")],
        stream=True,
    )

    chunks = []
    async for chunk in provider.stream(request):
        chunks.append(chunk)

    assert len(chunks) > 1
    assert any(c.choices[0].delta.content for c in chunks if c.choices[0].delta.content)
    assert chunks[-1].choices[0].finish_reason == "stop"

    await provider.close()


@pytest.mark.asyncio
async def test_provider_rejects_paid_model_on_free_only():
    provider = MockOpenRouterProvider()

    request = ChatCompletionRequest(
        model="openai/gpt-4o",
        messages=[ChatMessage(role="user", content="Should fail")],
    )

    with pytest.raises(FreeTierViolationError):
        await provider.complete(request, model_name="openai/gpt-4o")

    await provider.close()
