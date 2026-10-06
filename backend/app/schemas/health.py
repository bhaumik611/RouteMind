from typing import Dict
from pydantic import BaseModel, Field


class ComponentHealth(BaseModel):
    status: str = Field(description="Component status: 'healthy', 'degraded', or 'unhealthy'")
    latency_ms: float = Field(default=0.0, description="Probe latency in milliseconds")
    details: Dict[str, str] = Field(default_factory=dict, description="Diagnostic details")


class HealthResponse(BaseModel):
    status: str = Field(description="Overall system health status: 'healthy', 'degraded', 'unhealthy'")
    version: str
    free_only_enforced: bool
    environment: str
    components: Dict[str, ComponentHealth] = Field(default_factory=dict)
