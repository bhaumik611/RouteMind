import time
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

from app.core.constants import RoutingStrategy


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function", "tool"] = Field(
        ..., description="The role of the message author."
    )
    content: str = Field(..., description="The content of the message.")
    name: Optional[str] = Field(None, description="Optional name of the participant.")


class ChatCompletionRequest(BaseModel):
    model: str = Field(
        default="auto",
        description="ID of the model or 'auto' for dynamic routing.",
    )
    messages: List[ChatMessage] = Field(
        ..., min_length=1, description="List of messages comprising the conversation."
    )
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1, le=1)
    stream: Optional[bool] = Field(default=False, description="Whether to stream back partial chunks.")
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=32768)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = Field(default=None, description="Optional end-user identifier.")

    # RouteMind Extensions
    routing_strategy: Optional[RoutingStrategy] = Field(
        default=None,
        description="Override the default routing strategy (e.g., bandit, rule_based, latency_first).",
    )
    tenant_id: Optional[str] = Field(
        default="default",
        description="Tenant identifier for multi-tenant rate limiting and policy enforcement.",
    )
    latency_sensitive: Optional[bool] = Field(
        default=False,
        description="Hint to router that minimal response latency should be prioritized.",
    )


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RouteMindMeta(BaseModel):
    request_id: str
    selected_provider: str
    selected_model: str
    strategy_applied: str
    latency_ms: float
    fallback_triggered: bool = False
    fallback_attempts: int = 0
    circuit_state: str = "CLOSED"
    estimated_cost_usd: float = 0.0


class ChatCompletionChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionChoiceMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo
    routemind_meta: RouteMindMeta


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChunkChoice]
    routemind_meta: Optional[RouteMindMeta] = None
