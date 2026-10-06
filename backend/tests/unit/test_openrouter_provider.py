import pytest
import respx
import httpx

from app.core.exceptions import FreeTierViolationError, ProviderUnavailableError
from app.providers.openrouter import OpenRouterProvider
from app.schemas.chat import ChatCompletionRequest, ChatMessage


@pytest.mark.asyncio
@respx.mock
async def test_openrouter_complete_success():
    provider = OpenRouterProvider()
    model = "meta-llama/llama-3-8b-instruct:free"

    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "gen-12345",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Distributed systems require consensus protocols like Raft.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 10,
                    "total_tokens": 25,
                },
            },
        )
    )

    request = ChatCompletionRequest(
        model=model,
        messages=[ChatMessage(role="user", content="Explain consensus")],
    )

    response = await provider.complete(request)
    assert response.provider.value == "openrouter"
    assert "Distributed systems require consensus" in response.content
    assert response.usage.total_tokens == 25
    assert response.latency_ms > 0

    await provider.close()


@pytest.mark.asyncio
@respx.mock
async def test_openrouter_rate_limit_429():
    provider = OpenRouterProvider()
    model = "meta-llama/llama-3-8b-instruct:free"

    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            429,
            json={"error": {"message": "Rate limit exceeded on free tier", "code": 429}},
        )
    )

    request = ChatCompletionRequest(
        model=model,
        messages=[ChatMessage(role="user", content="Hello")],
    )

    with pytest.raises(ProviderUnavailableError) as exc:
        await provider.complete(request)

    assert "HTTP 429" in str(exc.value)

    await provider.close()


@pytest.mark.asyncio
async def test_openrouter_rejects_paid_model():
    provider = OpenRouterProvider()
    request = ChatCompletionRequest(
        model="openai/gpt-4o",
        messages=[ChatMessage(role="user", content="Paid model request")],
    )

    with pytest.raises(FreeTierViolationError):
        await provider.complete(request)

    await provider.close()
