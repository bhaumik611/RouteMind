import itertools
import threading
from typing import Dict, List

from app.core.constants import RoutingStrategy
from app.routing.base import BaseRouter
from app.schemas.chat import ChatCompletionRequest
from app.schemas.features import RequestFeatures
from app.schemas.provider import ModelMetadata


class RoundRobinRouter(BaseRouter):
    """Thread-safe cyclic round-robin selection across eligible candidates."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counter = 0

    @property
    def strategy_name(self) -> RoutingStrategy:
        return RoutingStrategy.ROUND_ROBIN

    def select_model(
        self,
        request: ChatCompletionRequest,
        features: RequestFeatures,
        candidates: List[ModelMetadata],
    ) -> ModelMetadata:
        self.validate_candidates(candidates)
        with self._lock:
            index = self._counter % len(candidates)
            self._counter += 1
            return candidates[index]
