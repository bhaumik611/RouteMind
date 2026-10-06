import pytest
from app.core.constants import TaskCategory
from app.routing.analyzer import RequestAnalyzer
from app.schemas.chat import ChatCompletionRequest, ChatMessage


def test_token_estimation_and_length():
    prompt = "Explain how distributed transactions work with two-phase commit."
    tokens = RequestAnalyzer.estimate_tokens(prompt)
    assert tokens > 5
    assert tokens < 50


def test_coding_task_detection():
    req = ChatCompletionRequest(
        model="auto",
        messages=[
            ChatMessage(role="user", content="Write a python function `def fibonacci(n):` using memoization.")
        ],
    )
    features = RequestAnalyzer.analyze(req)
    assert features.requires_code is True
    assert features.task_category == TaskCategory.CODING
    assert features.prompt_length_chars > 20


def test_reasoning_task_detection():
    req = ChatCompletionRequest(
        model="auto",
        messages=[
            ChatMessage(role="user", content="Step-by-step explain why CAP theorem forbids CA under network partitions.")
        ],
    )
    features = RequestAnalyzer.analyze(req)
    assert features.requires_reasoning is True
    assert features.task_category == TaskCategory.REASONING


def test_summarization_detection():
    req = ChatCompletionRequest(
        model="auto",
        messages=[
            ChatMessage(role="user", content="Please summarize the following quarterly report in 3 bullet points.")
        ],
    )
    features = RequestAnalyzer.analyze(req)
    assert features.task_category == TaskCategory.SUMMARIZATION


def test_feature_vector_serialization():
    req = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="Hello world")],
    )
    features = RequestAnalyzer.analyze(req)
    vec = features.to_feature_vector()
    assert len(vec) == 10
    assert vec[0] == 1.0  # Bias term
