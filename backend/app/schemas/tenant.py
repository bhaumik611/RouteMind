from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

from app.core.constants import RoutingStrategy, UserRole


class TenantBase(BaseModel):
    id: str = Field(..., min_length=2, max_length=64, description="Unique alphanumeric tenant slug")
    name: str = Field(..., min_length=2, max_length=255, description="Tenant organization display name")


class TenantCreate(TenantBase):
    daily_token_budget: Optional[int] = Field(default=1000000)
    daily_request_budget: Optional[int] = Field(default=5000)
    default_strategy: Optional[RoutingStrategy] = Field(default=RoutingStrategy.BANDIT)


class TenantPolicySchema(BaseModel):
    daily_token_budget: int
    daily_request_budget: int
    allowed_strategies: List[str]
    default_strategy: str
    enforce_free_only: bool


class TenantResponse(TenantBase):
    is_active: bool
    created_at: datetime
    policy: Optional[TenantPolicySchema] = None


class APIKeyCreate(BaseModel):
    tenant_id: str
    role: UserRole = UserRole.DEVELOPER
    rate_limit_rpm: int = 60
    expires_in_days: Optional[int] = 365


class APIKeyResponse(BaseModel):
    id: str
    tenant_id: str
    key_prefix: str
    role: str
    rate_limit_rpm: int
    is_active: bool
    plain_key: Optional[str] = None  # Returned once on creation only
    created_at: datetime
