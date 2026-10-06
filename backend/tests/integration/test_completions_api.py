import pytest
from httpx import ASGITransport, AsyncClient
from app.core.constants import HEADER_LATENCY_MS, HEADER_REQUEST_ID
from app.main import app


@pytest.mark.asyncio
async def test_chat_completions_auto_routing():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "model": "auto",
            "messages": [
                {"role": "user", "content": "Explain how database indexing works with B-trees."}
            ],
            "routing_strategy": "rule_based",
        }
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(data["choices"][0]["message"]["content"]) > 0

    # Validate RouteMind metadata
    meta = data["routemind_meta"]
    assert meta["strategy_applied"] == "rule_based"
    assert meta["estimated_cost_usd"] == 0.0
    assert meta["latency_ms"] > 0
    assert HEADER_REQUEST_ID in response.headers
    assert HEADER_LATENCY_MS in response.headers


@pytest.mark.asyncio
async def test_chat_completions_streaming():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "model": "auto",
            "messages": [{"role": "user", "content": "Stream me a response."}],
            "stream": True,
        }
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    text = response.text
    assert "data:" in text
    assert "[DONE]" in text
