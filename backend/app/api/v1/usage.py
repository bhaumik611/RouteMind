from typing import Any, Dict
from fastapi import APIRouter, Query, status

from app.services.quota_manager import quota_manager

router = APIRouter()


@router.get(
    "/usage",
    status_code=status.HTTP_200_OK,
    summary="Get tenant usage and quota metrics",
    description="Returns current token consumption, request counts, and remaining daily virtual budget for a tenant.",
)
async def get_usage(
    tenant_id: str = Query(default="default", description="Tenant identifier to inspect")
) -> Dict[str, Any]:
    return await quota_manager.get_tenant_usage(tenant_id=tenant_id)
