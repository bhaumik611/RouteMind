from datetime import datetime, timezone
from sqlalchemy import ARRAY, BigInteger, Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.core.database import Base


class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(String(64), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    policy = relationship("TenantPolicy", back_populates="tenant", uselist=False, cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="tenant", cascade="all, delete-orphan")


class TenantPolicy(Base):
    __tablename__ = "tenant_policies"

    tenant_id = Column(String(64), ForeignKey("tenants.id", ondelete="CASCADE"), primary_key=True)
    daily_token_budget = Column(BigInteger, default=1000000, nullable=False)
    daily_request_budget = Column(Integer, default=5000, nullable=False)
    allowed_strategies = Column(String(255), default="bandit,rule_based,latency_first,cheapest", nullable=False)
    default_strategy = Column(String(32), default="bandit", nullable=False)
    enforce_free_only = Column(Boolean, default=True, nullable=False)

    tenant = relationship("Tenant", back_populates="policy")


class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(String(64), primary_key=True, index=True)
    tenant_id = Column(String(64), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    hashed_key = Column(String(128), unique=True, nullable=False, index=True)
    key_prefix = Column(String(16), nullable=False)
    role = Column(String(32), default="DEVELOPER", nullable=False)
    rate_limit_rpm = Column(Integer, default=60, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)

    tenant = relationship("Tenant", back_populates="api_keys")
