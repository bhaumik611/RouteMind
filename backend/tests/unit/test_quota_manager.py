import pytest
from app.core.exceptions import QuotaExceededError
from app.services.quota_manager import VirtualQuotaManager


@pytest.mark.asyncio
async def test_quota_manager_tracks_and_exceeds():
    qm = VirtualQuotaManager(redis_url="")
    tenant = "test-quota-tenant"

    # Initial check passes
    await qm.check_quota(tenant, estimated_tokens=500, daily_token_limit=1000, daily_request_limit=10)

    # Record 800 tokens
    await qm.record_usage(tenant, tokens=800)

    usage = await qm.get_tenant_usage(tenant)
    assert usage["tokens_consumed_today"] == 800
    assert usage["requests_today"] == 1

    # Next check requesting 300 tokens should fail (800 + 300 > 1000)
    with pytest.raises(QuotaExceededError) as exc:
        await qm.check_quota(tenant, estimated_tokens=300, daily_token_limit=1000, daily_request_limit=10)

    assert exc.value.status_code == 429
