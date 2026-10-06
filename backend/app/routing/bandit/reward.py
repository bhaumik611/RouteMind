import math
from typing import Optional
from app.schemas.features import RequestFeatures
from app.schemas.provider import ProviderResponse


class MultiObjectiveReward:
    """Computes normalized multi-objective utility reward for bandit online learning.
    Reward = α * Quality - β * Normalized_Latency - γ * Resource_Proxy - δ * Failure_Penalty
    """

    def __init__(
        self,
        alpha: float = 1.0,  # Quality weight
        beta: float = 0.3,   # Latency weight
        gamma: float = 0.2,  # Resource cost proxy weight
        delta: float = 2.0,  # Failure penalty
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def estimate_quality(
        self, features: RequestFeatures, response: Optional[ProviderResponse]
    ) -> float:
        """Estimates output quality score [0.0, 1.0] using deterministic criteria."""
        if not response or not response.content:
            return 0.0

        content = response.content.strip()
        if len(content) == 0:
            return 0.0

        score = 0.5  # Base score for non-empty response

        # 1. Completeness heuristic
        word_count = len(content.split())
        if word_count > 10:
            score += 0.2
        if word_count > 50:
            score += 0.1

        # 2. Task-specific quality checks
        if features.requires_code:
            if "```" in content or "def " in content or "function" in content:
                score += 0.2
            else:
                score -= 0.1

        if features.requires_structured_output:
            if "{" in content and "}" in content:
                score += 0.2

        if features.requires_reasoning:
            if any(term in content.lower() for term in ["because", "therefore", "step", "first"]):
                score += 0.1

        return min(1.0, max(0.0, score))

    def compute_reward(
        self,
        features: RequestFeatures,
        response: Optional[ProviderResponse],
        latency_ms: float,
        is_success: bool = True,
    ) -> float:
        """Calculates total scalar reward for online bandit updates."""
        if not is_success or not response:
            return -self.delta

        quality = self.estimate_quality(features, response)

        # Normalize latency (0ms -> 0.0, 1000ms -> 0.5, 3000ms+ -> 1.0)
        norm_latency = min(1.0, math.log1p(latency_ms) / math.log1p(3000.0))

        # Token footprint / resource proxy (normalized 0.0 to 1.0)
        total_tokens = response.usage.total_tokens if response.usage else 50
        norm_resource = min(1.0, total_tokens / 2048.0)

        reward = (
            self.alpha * quality
            - self.beta * norm_latency
            - self.gamma * norm_resource
        )

        return round(reward, 4)


reward_evaluator = MultiObjectiveReward()
