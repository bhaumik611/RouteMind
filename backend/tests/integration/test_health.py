import pytest
from httpx import ASGITransport, AsyncClient
from app.core.constants import HEADER_LATENCY_MS, HEADER_REQUEST_ID
from app.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["free_only_enforced"] is True
    assert "gateway" in data["components"]
    assert "safety_guard" in data["components"]

    # Verify correlation & latency middleware headers
    assert HEADER_REQUEST_ID in response.headers
    assert HEADER_LATENCY_MS in response.headers


@pytest.mark.asyncio
async def test_liveness_and_readiness():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        live_res = await client.get("/v1/health/live")
        ready_res = await client.get("/v1/health/ready")

    assert live_res.status_code == 200
    assert live_res.json()["status"] == "alive"

    assert ready_res.status_code == 200
    assert ready_res.json()["status"] == "ready"
    assert ready_res.json()["free_only"] is True


@pytest.mark.asyncio
async def test_custom_request_id_preservation():
    custom_req_id = "test-custom-trace-id-12345"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/health", headers={HEADER_REQUEST_ID: custom_req_id})

    assert response.status_code == 200
    assert response.headers[HEADER_REQUEST_ID] == custom_req_id
