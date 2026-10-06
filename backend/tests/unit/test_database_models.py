from app.models.bandit_state import BanditModelState
from app.models.provider import ModelCatalog, ProviderModel
from app.models.request_log import RequestLog
from app.models.tenant import APIKey, Tenant, TenantPolicy


def test_tenant_and_policy_model():
    tenant = Tenant(id="tenant-alpha", name="Alpha Corp")
    policy = TenantPolicy(
        tenant_id="tenant-alpha",
        daily_token_budget=500000,
        daily_request_budget=2000,
        enforce_free_only=True,
    )
    api_key = APIKey(
        id="key-1",
        tenant_id="tenant-alpha",
        hashed_key="hash123",
        key_prefix="rm-live",
        role="DEVELOPER",
    )

    assert tenant.id == "tenant-alpha"
    assert policy.daily_token_budget == 500000
    assert policy.enforce_free_only is True
    assert api_key.role == "DEVELOPER"


def test_request_log_model():
    log = RequestLog(
        request_id="req-999",
        tenant_id="tenant-alpha",
        selected_provider_id="gemini",
        selected_model_id="gemini/gemini-1.5-flash",
        routing_strategy="bandit",
        routing_features={"tokens": 150, "category": "coding"},
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        latency_ms=230.5,
        status_code=200,
        is_fallback=False,
    )

    assert log.request_id == "req-999"
    assert log.selected_provider_id == "gemini"
    assert log.total_tokens == 150
    assert log.latency_ms == 230.5


def test_bandit_model_state():
    state = BanditModelState(
        model_id="openrouter/meta-llama/llama-3-8b-instruct:free",
        strategy="linucb",
        feature_dimension=10,
        matrix_a_json=[[1.0] * 10 for _ in range(10)],
        vector_b_json=[0.0] * 10,
        total_pulls=42,
        cumulative_reward=35.5,
    )

    assert state.model_id == "openrouter/meta-llama/llama-3-8b-instruct:free"
    assert state.total_pulls == 42
    assert state.cumulative_reward == 35.5
