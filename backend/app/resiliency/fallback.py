from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.core.config import settings
from app.core.constants import ProviderType
from app.core.exceptions import FreeTierViolationError, ProviderUnavailableError
from app.core.logging import logger
from app.providers.base import BaseLLMProvider
from app.providers.registry import registry
from app.resiliency.circuit_breaker import circuit_breaker
from app.resiliency.retry import RetryPolicy
from app.schemas.chat import ChatCompletionRequest
from app.schemas.provider import ModelMetadata, ProviderResponse


@dataclass
class FallbackResult:
    response: ProviderResponse
    model: ModelMetadata
    provider: BaseLLMProvider
    is_fallback: bool
    fallback_attempts: int
    circuit_state: str


class FallbackEngine:
    """Orchestrates resilient multi-provider escalation, circuit checks, and failovers."""

    def __init__(self, retry_policy: Optional[RetryPolicy] = None):
        self.retry_policy = retry_policy or RetryPolicy(max_retries=1, initial_delay_seconds=0.05)

    def order_fallback_candidates(
        self,
        primary_model: ModelMetadata,
        all_candidates: List[ModelMetadata],
    ) -> List[ModelMetadata]:
        """Builds an ordered priority list of fallback candidates excluding the primary model."""
        ordered: List[ModelMetadata] = [primary_model]

        # Sort remaining candidates by quality score descending
        remaining = [m for m in all_candidates if m.id != primary_model.id]
        remaining.sort(key=lambda m: m.quality_score, reverse=True)
        ordered.extend(remaining)

        # Ensure local mock is present as ultimate safety net
        if not any(m.provider_id == ProviderType.LOCAL_MOCK for m in ordered):
            local_models = [m for m in registry.list_models(free_only=True) if m.provider_id == ProviderType.LOCAL_MOCK]
            if local_models:
                ordered.append(local_models[0])

        return ordered

    async def execute_resiliently(
        self,
        request: ChatCompletionRequest,
        primary_model: ModelMetadata,
        candidate_pool: Optional[List[ModelMetadata]] = None,
    ) -> FallbackResult:
        """Executes completion with automatic circuit breaker checks, retry policy, and multi-model fallback."""
        all_candidates = candidate_pool or registry.list_models(free_only=settings.FREE_ONLY)
        fallback_chain = self.order_fallback_candidates(primary_model, all_candidates)

        attempts = 0
        last_error: Optional[Exception] = None

        for idx, candidate in enumerate(fallback_chain):
            provider_id_str = candidate.provider_id.value
            provider = registry.get(candidate.provider_id)

            # 1. Circuit Breaker Check
            if not circuit_breaker.can_execute(provider_id_str):
                logger.warning(
                    f"Skipping [{candidate.id}] because circuit breaker for [{provider_id_str}] is OPEN.",
                    extra={"extra_data": {"provider": provider_id_str, "model": candidate.id}},
                )
                continue

            attempts += 1
            is_fallback = idx > 0

            # 2. Attempt execution with retry
            try:
                async def _call():
                    return await provider.complete(request, model_name=candidate.model_name)

                response = await self.retry_policy.execute_with_retry(
                    _call, provider_name=provider.name
                )

                # Record success in Circuit Breaker
                circuit_breaker.record_success(provider_id_str)

                if is_fallback:
                    logger.info(
                        f"Fallback succeeded on attempt {attempts} using model [{candidate.id}].",
                        extra={"extra_data": {"fallback_model": candidate.id, "attempts": attempts}},
                    )

                return FallbackResult(
                    response=response,
                    model=candidate,
                    provider=provider,
                    is_fallback=is_fallback,
                    fallback_attempts=attempts - 1,
                    circuit_state=circuit_breaker.get_state(provider_id_str).value,
                )

            except FreeTierViolationError:
                raise
            except Exception as e:
                last_error = e
                # Record failure in Circuit Breaker
                circuit_breaker.record_failure(provider_id_str)
                logger.warning(
                    f"Provider [{provider.name}] failed for model [{candidate.id}]: {str(e)}. Escalating to next fallback arm.",
                    extra={"extra_data": {"provider": provider_id_str, "model": candidate.id, "error": str(e)}},
                )

        # All fallback arms exhausted
        raise ProviderUnavailableError(
            f"All {len(fallback_chain)} candidate providers/models in the fallback matrix failed. Last error: {str(last_error)}"
        )


# Global Fallback Engine Singleton
fallback_engine = FallbackEngine()
