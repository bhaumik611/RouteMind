from typing import List
from fastapi import APIRouter, Query, status

from app.core.config import settings
from app.providers.registry import registry
from app.schemas.provider import ModelMetadata

router = APIRouter()


@router.get(
    "/models",
    response_model=List[ModelMetadata],
    status_code=status.HTTP_200_OK,
    summary="List available models",
    description="Returns the active catalog of registered models. Strictly free-tier only when FREE_ONLY=True.",
)
async def list_models(
    free_only: bool = Query(default=True, description="Filter only zero-cost models")
) -> List[ModelMetadata]:
    effective_free_only = True if settings.FREE_ONLY else free_only
    return registry.list_models(free_only=effective_free_only)
