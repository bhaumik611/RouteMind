from typing import List

from app.core.constants import RoutingStrategy
from app.routing.base import BaseRouter
from app.schemas.chat import ChatCompletionRequest
from app.schemas.features import RequestFeatures
from app.schemas.provider import ModelMetadata


class LatencyFirstRouter(BaseRouter):
    """Selects the candidate with the lowest expected / historical latency."""

    @property
    def strategy_name(self) -> RoutingStrategy:
        return RoutingStrategy.LATENCY_FIRST

    def select_model(
        self,
        request: ChatCompletionRequest,
        features: RequestFeatures,
        candidates: List[ModelMetadata],
    ) -> ModelMetadata:
        self.validate_candidates(candidates)
        return min(candidates, key=lambda m: m.expected_latency_ms)
