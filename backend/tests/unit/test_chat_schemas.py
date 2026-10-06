import pytest
from pydantic import ValidationError

from app.core.constants import RoutingStrategy
from app.schemas.chat import (
    ChatCompletionRequest,
    ChatMessage,
    ChatCompletionResponse,
    RouteMindMeta,
    UsageInfo,
)


def test_valid_chat_completion_request():
    req = ChatCompletionRequest(
        model="auto",
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Explain Paxos consensus."),
        ],
        temperature=0.5,
        routing_strategy=RoutingStrategy.BANDIT,
    )
    assert req.model == "auto"
    assert len(req.messages) == 2
    assert req.routing_strategy == RoutingStrategy.BANDIT


def test_empty_messages_validation():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(messages=[])


def test_response_schema_structure():
    res = ChatCompletionResponse(
        id="chatcmpl-test-123",
        model="local_mock/mock-free-model",
        choices=[],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        routemind_meta=RouteMindMeta(
            request_id="req-123",
            selected_provider="local_mock",
            selected_model="mock-free-model",
            strategy_applied="bandit",
            latency_ms=45.2,
        ),
    )
    assert res.usage.total_tokens == 30
    assert res.routemind_meta.latency_ms == 45.2
