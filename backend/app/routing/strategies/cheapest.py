from typing import List

from app.core.constants import RoutingStrategy
from app.routing.base import BaseRouter
from app.schemas.chat import ChatCompletionRequest
from app.schemas.features import RequestFeatures
from app.schemas.provider import ModelMetadata


class CheapestRouter(BaseRouter):
    """Selects the candidate with lowest token cost / minimal resource footprint proxy."""

    @property
    def strategy_name(self) -> RoutingStrategy:
        return RoutingStrategy.CHEAPEST

    def select_model(
        self,
        request: ChatCompletionRequest,
        features: RequestFeatures,
        candidates: List[ModelMetadata],
    ) -> ModelMetadata:
        self.validate_candidates(candidates)
        # All models have 0.00 price, so we sort primarily by cost, then by smaller context window/footprint
        return min(candidates, key=lambda m: (m.cost_per_1k_input_usd, m.context_window))
