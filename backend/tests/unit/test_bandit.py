import pytest
from app.routing.bandit.linucb import DisjointLinUCB, LinUCBArm
from app.routing.bandit.reward import MultiObjectiveReward
from app.schemas.chat import UsageInfo
from app.schemas.features import RequestFeatures
from app.schemas.provider import ProviderResponse
from app.core.constants import ProviderType


def test_linucb_arm_initialization():
    arm = LinUCBArm(arm_id="test-arm", dimension=10, alpha=1.0)
    assert arm.d == 10
    assert arm.pulls == 0
    assert arm.total_reward == 0.0

    # Utility score before any updates
    x = [1.0] + [0.0] * 9
    ucb, expected, bonus = arm.compute_utility(x)
    assert expected == 0.0
    assert bonus > 0.0  # Exploration bonus is non-zero initially


def test_linucb_online_update_and_preference():
    bandit = DisjointLinUCB(dimension=10, alpha=0.5)
    arm1 = "model-alpha"
    arm2 = "model-beta"

    x = [1.0, 0.5, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Initially both have similar exploratory scores
    best_arm, _, _, _ = bandit.select_arm([arm1, arm2], x)

    # Train arm1 with high rewards repeatedly
    for _ in range(10):
        bandit.update_arm(arm1, x, reward=1.0)
        bandit.update_arm(arm2, x, reward=-0.5)

    # Bandit should now strongly prefer arm1
    best_arm, score, expected, _ = bandit.select_arm([arm1, arm2], x)
    assert best_arm == arm1
    assert expected > 0.0


def test_multi_objective_reward_calculation():
    evaluator = MultiObjectiveReward(alpha=1.0, beta=0.3, gamma=0.2, delta=2.0)
    features = RequestFeatures(requires_code=True)
    response = ProviderResponse(
        content="```python\ndef solve(): pass\n```",
        model="mock-coding-model",
        provider=ProviderType.LOCAL_MOCK,
        usage=UsageInfo(prompt_tokens=20, completion_tokens=30, total_tokens=50),
    )

    reward = evaluator.compute_reward(features, response, latency_ms=100.0, is_success=True)
    assert reward > 0.0  # Quality exceeds small latency/resource penalty

    # Failure penalty
    fail_reward = evaluator.compute_reward(features, None, latency_ms=500.0, is_success=False)
    assert fail_reward == -2.0
