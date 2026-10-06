import pytest
import respx
import httpx
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.resiliency.circuit_breaker import circuit_breaker


@pytest.mark.asyncio
@respx.mock
async def test_resilient_fallback_on_provider_outage():
    # 1. Reset circuit breaker for clean test
    circuit_breaker.reset("openrouter")
    circuit_breaker.reset("gemini")

    # 2. Mock OpenRouter to fail with 500 error
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "Internal OpenRouter Outage"})
    )

    # 3. Send completion request targeting auto
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "model": "auto",
            "messages": [
                {"role": "user", "content": "Explain fault tolerance."}
            ],
            "routing_strategy": "round_robin",
        }
        response = await client.post("/v1/chat/completions", json=payload)

    # The request should succeed despite OpenRouter 500 outage due to fallback engine
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["content"] is not None

    meta = data["routemind_meta"]
    # Fallback or successful candidate used
    assert meta["selected_provider"] in ("local_mock", "gemini", "openrouter")
