import uuid
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from app.core.constants import UserRole
from app.core.database import async_session_factory
from app.models.tenant import APIKey, Tenant, TenantPolicy
from app.schemas.tenant import APIKeyCreate, APIKeyResponse, TenantCreate, TenantResponse
from app.security.auth import AuthenticatedContext, require_role
from app.security.hashing import generate_api_key, hash_api_key

router = APIRouter()


@router.post(
    "/admin/tenants",
    response_model=TenantResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Provision a new tenant organization",
    dependencies=[Depends(require_role([UserRole.ADMIN, UserRole.DEVELOPER]))],
)
async def create_tenant(payload: TenantCreate) -> Any:
    async with async_session_factory() as session:
        # Check uniqueness
        stmt = select(Tenant).where(Tenant.id == payload.id)
        existing = (await session.execute(stmt)).scalars().first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Tenant ID '{payload.id}' already exists.")

        tenant = Tenant(id=payload.id, name=payload.name, is_active=True)
        policy = TenantPolicy(
            tenant_id=payload.id,
            daily_token_budget=payload.daily_token_budget or 1000000,
            daily_request_budget=payload.daily_request_budget or 5000,
            default_strategy=payload.default_strategy.value if payload.default_strategy else "bandit",
            enforce_free_only=True,
        )
        session.add(tenant)
        session.add(policy)
        await session.commit()

        return TenantResponse(
            id=tenant.id,
            name=tenant.name,
            is_active=tenant.is_active,
            created_at=tenant.created_at,
            policy={
                "daily_token_budget": policy.daily_token_budget,
                "daily_request_budget": policy.daily_request_budget,
                "allowed_strategies": policy.allowed_strategies.split(","),
                "default_strategy": policy.default_strategy,
                "enforce_free_only": policy.enforce_free_only,
            },
        )


@router.post(
    "/admin/keys",
    response_model=APIKeyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a scoped API Key for a tenant",
    dependencies=[Depends(require_role([UserRole.ADMIN, UserRole.DEVELOPER]))],
)
async def create_api_key(payload: APIKeyCreate) -> Any:
    plain_key = generate_api_key(prefix="rm-live")
    hashed = hash_api_key(plain_key)
    key_id = f"key_{uuid.uuid4().hex[:12]}"
    prefix = plain_key[:12]

    async with async_session_factory() as session:
        # Verify tenant exists
        stmt = select(Tenant).where(Tenant.id == payload.tenant_id)
        tenant = (await session.execute(stmt)).scalars().first()
        if not tenant:
            # Auto-create tenant if missing
            tenant = Tenant(id=payload.tenant_id, name=payload.tenant_id.capitalize())
            session.add(tenant)

        api_key = APIKey(
            id=key_id,
            tenant_id=payload.tenant_id,
            hashed_key=hashed,
            key_prefix=prefix,
            role=payload.role.value,
            rate_limit_rpm=payload.rate_limit_rpm,
            is_active=True,
        )
        session.add(api_key)
        await session.commit()

        return APIKeyResponse(
            id=api_key.id,
            tenant_id=api_key.tenant_id,
            key_prefix=api_key.key_prefix,
            role=api_key.role,
            rate_limit_rpm=api_key.rate_limit_rpm,
            is_active=api_key.is_active,
            plain_key=plain_key,  # Sent only once
            created_at=api_key.created_at,
        )
