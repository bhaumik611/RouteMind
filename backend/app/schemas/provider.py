from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.core.constants import ProviderType
from app.schemas.chat import UsageInfo


class ModelMetadata(BaseModel):
    id: str = Field(..., description="Unique model identifier (e.g., 'openrouter/meta-llama/llama-3-8b-instruct:free')")
    provider_id: ProviderType = Field(..., description="Provider hosting this model")
    model_name: str = Field(..., description="Upstream model name string")
    is_free: bool = Field(default=True, description="Strictly true for zero-cost free models")
    context_window: int = Field(default=8192, description="Maximum tokens supported")
    supports_streaming: bool = Field(default=True)
    supports_tools: bool = Field(default=False)
    cost_per_1k_input_usd: float = Field(default=0.0)
    cost_per_1k_output_usd: float = Field(default=0.0)
    quality_score: float = Field(default=0.8, description="Estimated baseline quality rating (0.0 to 1.0)")
    expected_latency_ms: float = Field(default=500.0, description="Baseline expected latency in milliseconds")
    is_active: bool = Field(default=True)


class ProviderMetadata(BaseModel):
    id: ProviderType
    name: str
    is_enabled: bool = True
    is_free_only_compatible: bool = True
    circuit_state: str = "CLOSED"
    models: List[ModelMetadata] = Field(default_factory=list)


class ProviderResponse(BaseModel):
    content: str
    model: str
    provider: ProviderType
    usage: UsageInfo
    raw_response: Optional[Dict[str, Any]] = None
    finish_reason: str = "stop"
    latency_ms: float = 0.0
