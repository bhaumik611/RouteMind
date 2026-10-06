import random
from typing import List

from app.core.constants import RoutingStrategy
from app.routing.base import BaseRouter
from app.schemas.chat import ChatCompletionRequest
from app.schemas.features import RequestFeatures
from app.schemas.provider import ModelMetadata


class RandomRouter(BaseRouter):
    """Uniform random candidate selection."""

    @property
    def strategy_name(self) -> RoutingStrategy:
        return RoutingStrategy.RANDOM

    def select_model(
        self,
        request: ChatCompletionRequest,
        features: RequestFeatures,
        candidates: List[ModelMetadata],
    ) -> ModelMetadata:
        self.validate_candidates(candidates)
        return random.choice(candidates)
