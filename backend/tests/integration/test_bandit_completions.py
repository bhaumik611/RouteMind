import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.routing.bandit.bandit_router import bandit_router


@pytest.mark.asyncio
async def test_bandit_completions_learning_loop():
    transport = ASGITransport(app=app)
    initial_pulls = sum(arm.pulls for arm in bandit_router.bandit.arms.values())

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Write a python unit test."}],
                "routing_strategy": "bandit",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["routemind_meta"]["strategy_applied"] == "bandit"

    # Verify arm pulls increased
    new_pulls = sum(arm.pulls for arm in bandit_router.bandit.arms.values())
    assert new_pulls > initial_pulls
