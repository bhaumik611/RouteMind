import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_routing_decision_introspection():
    transport = ASGITransport(app=app)
    custom_request_id = "req-audit-test-9999"

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Dispatch completion request with explicit request ID
        comp_res = await client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Explain binary search trees."}],
                "routing_strategy": "latency_first",
            },
            headers={"X-Request-ID": custom_request_id},
        )
        assert comp_res.status_code == 200

        # 2. Query routing decision introspection endpoint
        decision_res = await client.get(f"/v1/routing/decision/{custom_request_id}")
        assert decision_res.status_code == 200
        data = decision_res.json()["data"]

        assert data["request_id"] == custom_request_id
        assert data["routing_strategy"] == "latency_first"
        assert "routing_features" in data
        assert data["latency_ms"] > 0
        assert data["total_tokens"] > 0


@pytest.mark.asyncio
async def test_routing_decision_not_found():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/v1/routing/decision/non-existent-request-id")
        assert res.status_code == 404
