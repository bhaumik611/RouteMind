from typing import Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.constants import RoutingStrategy
from app.core.exceptions import ConfigurationError, ProviderUnavailableError
from app.core.logging import logger
from app.providers.registry import registry
from app.routing.analyzer import RequestAnalyzer
from app.routing.bandit.bandit_router import bandit_router
from app.routing.base import BaseRouter
from app.routing.strategies.cheapest import CheapestRouter
from app.routing.strategies.latency_first import LatencyFirstRouter
from app.routing.strategies.quality_first import QualityFirstRouter
from app.routing.strategies.random_router import RandomRouter
from app.routing.strategies.round_robin import RoundRobinRouter
from app.routing.strategies.rule_based import RuleBasedRouter
from app.schemas.chat import ChatCompletionRequest
from app.schemas.features import RequestFeatures
from app.schemas.provider import ModelMetadata


class RouterEngine:
    """Central Routing Engine orchestrating feature analysis, bandit learning, and model selection."""

    def __init__(self):
        self._strategies: Dict[RoutingStrategy, BaseRouter] = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        self.register_strategy(RandomRouter())
        self.register_strategy(RoundRobinRouter())
        self.register_strategy(CheapestRouter())
        self.register_strategy(LatencyFirstRouter())
        self.register_strategy(QualityFirstRouter())
        self.register_strategy(RuleBasedRouter())
        self.register_strategy(bandit_router)

    def register_strategy(self, router: BaseRouter) -> None:
        self._strategies[router.strategy_name] = router

    def get_strategy(self, strategy_name: RoutingStrategy) -> BaseRouter:
        if strategy_name not in self._strategies:
            return self._strategies.get(RoutingStrategy.BANDIT, self._strategies[RoutingStrategy.ROUND_ROBIN])
        return self._strategies[strategy_name]

    def route(
        self,
        request: ChatCompletionRequest,
        candidate_models: Optional[List[ModelMetadata]] = None,
    ) -> Tuple[ModelMetadata, RequestFeatures, str]:
        """Analyzes the incoming request and dispatches to the selected routing strategy."""
        features = RequestAnalyzer.analyze(request)
        candidates = candidate_models or registry.list_models(free_only=settings.FREE_ONLY)
        if not candidates:
            raise ProviderUnavailableError("No active free-tier models available for routing.")

        chosen_strategy_type = request.routing_strategy or settings.DEFAULT_ROUTING_STRATEGY
        router = self.get_strategy(chosen_strategy_type)
        selected_model = router.select_model(request, features, candidates)

        logger.info(
            f"Routed request to model [{selected_model.id}] via strategy [{router.strategy_name.value}]",
            extra={
                "extra_data": {
                    "strategy": router.strategy_name.value,
                    "model": selected_model.id,
                    "provider": selected_model.provider_id.value,
                    "category": features.task_category.value,
                    "tokens": features.estimated_prompt_tokens,
                }
            },
        )

        return selected_model, features, router.strategy_name.value


# Global Router Engine Singleton
router_engine = RouterEngine()
