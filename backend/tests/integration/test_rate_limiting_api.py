import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.services.quota_manager import quota_manager


@pytest.mark.asyncio
async def test_get_usage_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/usage?tenant_id=test-acme-tenant")

    assert response.status_code == 200
    data = response.json()
    assert data["tenant_id"] == "test-acme-tenant"
    assert "tokens_consumed_today" in data
    assert "token_budget_remaining" in data


@pytest.mark.asyncio
async def test_rate_limiting_on_completions():
    # Use a unique tenant ID for this test
    tenant = "rate-limit-test-tenant"
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First request should pass
        res1 = await client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "Hello"}]},
            headers={"X-Tenant-ID": tenant},
        )
        assert res1.status_code == 200

        # Check usage is recorded
        usage_res = await client.get(f"/v1/usage?tenant_id={tenant}")
        assert usage_res.status_code == 200
        assert usage_res.json()["requests_today"] >= 1
