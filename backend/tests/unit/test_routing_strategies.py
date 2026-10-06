import pytest
from app.core.constants import ProviderType, RoutingStrategy, TaskCategory
from app.routing.analyzer import RequestAnalyzer
from app.routing.strategies.cheapest import CheapestRouter
from app.routing.strategies.latency_first import LatencyFirstRouter
from app.routing.strategies.quality_first import QualityFirstRouter
from app.routing.strategies.random_router import RandomRouter
from app.routing.strategies.round_robin import RoundRobinRouter
from app.routing.strategies.rule_based import RuleBasedRouter
from app.schemas.chat import ChatCompletionRequest, ChatMessage
from app.schemas.features import RequestFeatures
from app.schemas.provider import ModelMetadata


@pytest.fixture
def sample_candidates():
    return [
        ModelMetadata(
            id="local/mock-fast",
            provider_id=ProviderType.LOCAL_MOCK,
            model_name="mock-fast-model",
            is_free=True,
            context_window=4096,
            quality_score=0.65,
            expected_latency_ms=60.0,
        ),
        ModelMetadata(
            id="local/mock-coding",
            provider_id=ProviderType.LOCAL_MOCK,
            model_name="mock-coding-model",
            is_free=True,
            context_window=16384,
            quality_score=0.88,
            expected_latency_ms=200.0,
        ),
        ModelMetadata(
            id="gemini/gemini-1.5-pro",
            provider_id=ProviderType.GEMINI,
            model_name="gemini-1.5-pro",
            is_free=True,
            context_window=1000000,
            quality_score=0.95,
            expected_latency_ms=800.0,
        ),
    ]


def test_latency_first_router(sample_candidates):
    router = LatencyFirstRouter()
    req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")])
    features = RequestFeatures()
    selected = router.select_model(req, features, sample_candidates)
    assert selected.model_name == "mock-fast-model"
    assert selected.expected_latency_ms == 60.0


def test_quality_first_router(sample_candidates):
    router = QualityFirstRouter()
    req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")])
    features = RequestFeatures()
    selected = router.select_model(req, features, sample_candidates)
    assert selected.model_name == "gemini-1.5-pro"
    assert selected.quality_score == 0.95


def test_round_robin_router(sample_candidates):
    router = RoundRobinRouter()
    req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")])
    features = RequestFeatures()

    picks = [router.select_model(req, features, sample_candidates).model_name for _ in range(6)]
    assert picks == [
        "mock-fast-model",
        "mock-coding-model",
        "gemini-1.5-pro",
        "mock-fast-model",
        "mock-coding-model",
        "gemini-1.5-pro",
    ]


def test_rule_based_router_coding(sample_candidates):
    router = RuleBasedRouter()
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="def sort_array(arr): pass")]
    )
    features = RequestAnalyzer.analyze(req)
    selected = router.select_model(req, features, sample_candidates)
    assert (
        "coding" in selected.model_name
        or "llama" in selected.model_name
        or "gemini" in selected.model_name
    )
