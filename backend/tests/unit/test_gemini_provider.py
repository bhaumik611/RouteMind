import pytest
import respx
import httpx

from app.core.exceptions import FreeTierViolationError, ProviderUnavailableError
from app.providers.gemini import GeminiProvider
from app.schemas.chat import ChatCompletionRequest, ChatMessage


@pytest.mark.asyncio
@respx.mock
async def test_gemini_complete_success():
    provider = GeminiProvider()
    model = "gemini-1.5-flash"

    # Mock the Gemini generateContent POST request
    respx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent").mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": "Gemini models use multimodal transformer architectures."
                                }
                            ],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 20,
                    "candidatesTokenCount": 15,
                    "totalTokenCount": 35,
                },
            },
        )
    )

    request = ChatCompletionRequest(
        model=model,
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="How do transformers work?"),
        ],
    )

    response = await provider.complete(request)
    assert response.provider.value == "gemini"
    assert "Gemini models use multimodal" in response.content
    assert response.usage.total_tokens == 35
    assert response.latency_ms > 0

    await provider.close()


@pytest.mark.asyncio
@respx.mock
async def test_gemini_rate_limit_429():
    provider = GeminiProvider()
    model = "gemini-1.5-flash"

    respx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent").mock(
        return_value=httpx.Response(
            429,
            json={"error": {"code": 429, "message": "Resource has been exhausted (e.g. check quota)."}},
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
async def test_gemini_rejects_paid_or_invalid_model():
    provider = GeminiProvider()
    request = ChatCompletionRequest(
        model="gemini-ultra-expensive-version",
        messages=[ChatMessage(role="user", content="Unverified model")],
    )

    with pytest.raises(FreeTierViolationError):
        await provider.complete(request)

    await provider.close()
