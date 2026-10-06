import abc
from typing import List

from app.core.constants import RoutingStrategy
from app.core.exceptions import ProviderUnavailableError
from app.schemas.chat import ChatCompletionRequest
from app.schemas.features import RequestFeatures
from app.schemas.provider import ModelMetadata


class BaseRouter(abc.ABC):
    """Abstract base class for all routing algorithms."""

    @property
    @abc.abstractmethod
    def strategy_name(self) -> RoutingStrategy:
        pass

    @abc.abstractmethod
    def select_model(
        self,
        request: ChatCompletionRequest,
        features: RequestFeatures,
        candidates: List[ModelMetadata],
    ) -> ModelMetadata:
        """Selects the optimal model from a list of eligible free-tier candidates."""
        pass

    def validate_candidates(self, candidates: List[ModelMetadata]) -> None:
        """Ensures at least one eligible candidate model is available."""
        if not candidates:
            raise ProviderUnavailableError("No active free-tier candidate models are available for routing.")
