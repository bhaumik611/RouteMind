from typing import List

from app.core.constants import RoutingStrategy
from app.routing.bandit.linucb import DisjointLinUCB
from app.routing.base import BaseRouter
from app.schemas.chat import ChatCompletionRequest
from app.schemas.features import RequestFeatures
from app.schemas.provider import ModelMetadata


class ContextualBanditRouter(BaseRouter):
    """Bayesian Contextual Bandit Router using Disjoint LinUCB."""

    def __init__(self, dimension: int = 10, alpha: float = 0.8):
        self.bandit = DisjointLinUCB(dimension=dimension, alpha=alpha)

    @property
    def strategy_name(self) -> RoutingStrategy:
        return RoutingStrategy.BANDIT

    def select_model(
        self,
        request: ChatCompletionRequest,
        features: RequestFeatures,
        candidates: List[ModelMetadata],
    ) -> ModelMetadata:
        self.validate_candidates(candidates)

        # 1. Candidate arm IDs
        candidate_ids = [m.id for m in candidates]
        feature_vector = features.to_feature_vector()

        # 2. Select optimal arm via LinUCB exploration-exploitation
        best_id, ucb_score, expected, bonus = self.bandit.select_arm(candidate_ids, feature_vector)

        # 3. Resolve selected ModelMetadata
        chosen_model = next((m for m in candidates if m.id == best_id), candidates[0])
        return chosen_model


bandit_router = ContextualBanditRouter()
