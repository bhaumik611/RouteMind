import time
import uuid
from typing import AsyncIterator
from fastapi import APIRouter, Header, Request, status
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.constants import HEADER_REQUEST_ID, HEADER_TENANT_ID, ProviderType
from app.core.exceptions import ModelNotFoundError, ProviderUnavailableError
from app.core.logging import logger
from app.providers.registry import registry
from app.resiliency.circuit_breaker import circuit_breaker
from app.resiliency.fallback import fallback_engine
from app.routing.analyzer import RequestAnalyzer
from app.routing.bandit.bandit_router import bandit_router
from app.routing.bandit.reward import reward_evaluator
from app.routing.engine import router_engine
from app.services.quota_manager import quota_manager
from app.services.rate_limiter import rate_limiter
from app.services.telemetry_worker import telemetry_worker
from app.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    RouteMindMeta,
)

router = APIRouter()


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    status_code=status.HTTP_200_OK,
    summary="OpenAI-compatible chat completion",
    description="Executes a chat completion with dynamic free-tier model routing, rate limiting, quota tracking, circuit breakers, and online contextual bandit learning.",
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    raw_request: Request,
    x_request_id: str = Header(default=None, alias=HEADER_REQUEST_ID),
    x_tenant_id: str = Header(default=None, alias=HEADER_TENANT_ID),
):
    req_id = x_request_id or getattr(raw_request.state, "request_id", str(uuid.uuid4()))
    tenant_id = x_tenant_id or request.tenant_id or "default"
    start_time = time.perf_counter()

    # 1. Enforce Rate Limiting (100 req/min per tenant)
    await rate_limiter.enforce(identifier=f"tenant:{tenant_id}", max_requests=100, window_seconds=60)

    # 2. Extract Features & Check Daily Virtual Budget
    features = RequestAnalyzer.analyze(request)
    await quota_manager.check_quota(tenant_id=tenant_id, estimated_tokens=features.estimated_prompt_tokens)

    # 3. Dynamic Model Routing or Explicit Target
    if request.model == "auto":
        primary_model, _, strategy_used = router_engine.route(request)
    else:
        target_model_name = request.model
        try:
            primary_model = registry.get_model(target_model_name)
        except ModelNotFoundError:
            local_models = [m for m in registry.list_models(free_only=True) if m.provider_id == ProviderType.LOCAL_MOCK]
            primary_model = local_models[0]
        strategy_used = "explicit_override"

    # 4. Handle Streaming vs Non-Streaming
    if request.stream:
        provider = registry.get(primary_model.provider_id)
        async def event_generator() -> AsyncIterator[str]:
            async for chunk in provider.stream(request, model_name=primary_model.model_name):
                chunk_json = chunk.model_dump_json()
                yield f"data: {chunk_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={HEADER_REQUEST_ID: req_id, HEADER_TENANT_ID: tenant_id},
        )

    # 5. Resilient Non-Streaming Execution via Fallback Engine
    result = await fallback_engine.execute_resiliently(
        request=request,
        primary_model=primary_model,
    )

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # 6. Online Bandit Reward Calculation & Model Update
    reward = reward_evaluator.compute_reward(
        features=features,
        response=result.response,
        latency_ms=duration_ms,
        is_success=True,
    )
    bandit_router.bandit.update_arm(
        arm_id=result.model.id,
        feature_vector=features.to_feature_vector(),
        reward=reward,
    )

    # 7. Record Token Consumption
    await quota_manager.record_usage(
        tenant_id=tenant_id,
        tokens=result.response.usage.total_tokens,
    )

    # 8. Non-blocking Telemetry Logging
    telemetry_worker.record_event_sync({
        "request_id": req_id,
        "tenant_id": tenant_id,
        "selected_provider_id": result.provider.provider_type.value,
        "selected_model_id": result.model.id,
        "routing_strategy": strategy_used,
        "routing_features": features.model_dump(),
        "prompt_tokens": result.response.usage.prompt_tokens,
        "completion_tokens": result.response.usage.completion_tokens,
        "total_tokens": result.response.usage.total_tokens,
        "latency_ms": duration_ms,
        "status_code": 200,
        "is_fallback": result.is_fallback,
        "fallback_attempts": result.fallback_attempts,
        "calculated_reward": reward,
    })

    return ChatCompletionResponse(
        id=f"chatcmpl-{req_id[:12]}",
        model=result.model.model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionChoiceMessage(
                    role="assistant",
                    content=result.response.content,
                ),
                finish_reason=result.response.finish_reason,
            )
        ],
        usage=result.response.usage,
        routemind_meta=RouteMindMeta(
            request_id=req_id,
            selected_provider=result.provider.provider_type.value,
            selected_model=result.model.model_name,
            strategy_applied=strategy_used,
            latency_ms=duration_ms,
            fallback_triggered=result.is_fallback,
            fallback_attempts=result.fallback_attempts,
            circuit_state=result.circuit_state,
            estimated_cost_usd=0.0,
        ),
    )
