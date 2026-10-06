from typing import List

from app.core.constants import RoutingStrategy
from app.routing.base import BaseRouter
from app.schemas.chat import ChatCompletionRequest
from app.schemas.features import RequestFeatures
from app.schemas.provider import ModelMetadata


class QualityFirstRouter(BaseRouter):
    """Selects the candidate with the highest benchmark quality rating."""

    @property
    def strategy_name(self) -> RoutingStrategy:
        return RoutingStrategy.QUALITY_FIRST

    def select_model(
        self,
        request: ChatCompletionRequest,
        features: RequestFeatures,
        candidates: List[ModelMetadata],
    ) -> ModelMetadata:
        self.validate_candidates(candidates)
        return max(candidates, key=lambda m: m.quality_score)
