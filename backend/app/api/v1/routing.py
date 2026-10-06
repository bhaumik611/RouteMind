from typing import Any, Dict
from fastapi import APIRouter, HTTPException, Path, status

from app.services.telemetry_worker import telemetry_worker

router = APIRouter()


@router.get(
    "/routing/decision/{request_id}",
    status_code=status.HTTP_200_OK,
    summary="Get detailed routing decision audit log",
    description="Returns the exact feature vector, candidate arms, selected model, latency, tokens, and fallback details for a specific request ID.",
)
async def get_routing_decision(
    request_id: str = Path(..., description="Unique request ID to introspect")
) -> Dict[str, Any]:
    event = await telemetry_worker.get_recent_event(request_id)
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing decision log for request_id '{request_id}' not found.",
        )
    return {"success": True, "data": event}
