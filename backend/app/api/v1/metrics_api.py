from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter()


@router.get(
    "/metrics",
    summary="Prometheus metrics exporter",
    description="Exposes real-time Prometheus formatted metrics for gateway throughput, latency percentiles, error rates, and bandit rewards.",
)
async def get_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
