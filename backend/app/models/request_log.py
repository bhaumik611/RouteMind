from datetime import datetime, timezone
import json
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.types import JSON

from app.core.database import Base


class RequestLog(Base):
    __tablename__ = "request_logs"

    request_id = Column(String(64), primary_key=True, index=True)
    tenant_id = Column(String(64), nullable=False, index=True, default="default")
    api_key_id = Column(String(64), nullable=True, index=True)
    selected_provider_id = Column(String(32), nullable=False, index=True)
    selected_model_id = Column(String(128), nullable=False, index=True)
    routing_strategy = Column(String(32), nullable=False, index=True)
    routing_features = Column(JSON, nullable=False)
    routing_score = Column(Float, nullable=True)
    prompt_tokens = Column(Integer, default=0, nullable=False)
    completion_tokens = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)
    latency_ms = Column(Float, default=0.0, nullable=False)
    status_code = Column(Integer, default=200, nullable=False)
    is_fallback = Column(Boolean, default=False, nullable=False)
    fallback_attempts = Column(Integer, default=0, nullable=False)
    error_message = Column(Text, nullable=True)
    quality_score = Column(Float, nullable=True)
    calculated_reward = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
