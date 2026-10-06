from typing import List
from fastapi import APIRouter, status

from app.providers.registry import registry
from app.resiliency.circuit_breaker import circuit_breaker
from app.schemas.provider import ProviderMetadata

router = APIRouter()


@router.get(
    "/providers",
    response_model=List[ProviderMetadata],
    status_code=status.HTTP_200_OK,
    summary="List active providers and models",
    description="Returns metadata, operational status, circuit breaker states, and supported models for all registered LLM providers.",
)
async def list_providers() -> List[ProviderMetadata]:
    providers = registry.list_providers()
    # Enrich with live circuit breaker states
    for p in providers:
        p.circuit_state = circuit_breaker.get_state(p.id.value).value
    return providers
