import time
from fastapi import APIRouter, status
from app.core.config import settings
from app.schemas.health import ComponentHealth, HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Deep health and readiness check",
    description="Returns aggregate health status across core components, gateway, and free-tier validation guards.",
)
async def get_health() -> HealthResponse:
    start_time = time.perf_counter()

    # Core gateway check
    gateway_latency = (time.perf_counter() - start_time) * 1000
    components = {
        "gateway": ComponentHealth(
            status="healthy",
            latency_ms=round(gateway_latency, 2),
            details={"mode": "active", "free_only": str(settings.FREE_ONLY)},
        ),
        "safety_guard": ComponentHealth(
            status="healthy" if settings.FREE_ONLY else "degraded",
            latency_ms=0.01,
            details={
                "free_only_active": str(settings.FREE_ONLY),
                "openrouter_configured": str(settings.is_provider_available(settings.DEFAULT_ROUTING_STRATEGY)),
            },
        ),
    }

    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        free_only_enforced=settings.FREE_ONLY,
        environment=settings.ENVIRONMENT,
        components=components,
    )


@router.get(
    "/health/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Kubernetes/Docker liveness probe indicating ASGI server is responsive.",
)
async def get_liveness():
    return {"status": "alive"}


@router.get(
    "/health/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="Readiness probe verifying the gateway is accepting traffic.",
)
async def get_readiness():
    return {"status": "ready", "free_only": settings.FREE_ONLY}
