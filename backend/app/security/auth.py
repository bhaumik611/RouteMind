from typing import List, Optional
from fastapi import Depends, Header, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select

from app.core.constants import HEADER_API_KEY, HEADER_TENANT_ID, UserRole
from app.core.database import async_session_factory
from app.core.exceptions import AuthenticationError, PolicyViolationError
from app.models.tenant import APIKey, Tenant
from app.security.hashing import hash_api_key

api_key_header_auth = APIKeyHeader(name=HEADER_API_KEY, auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)


class AuthenticatedContext:
    def __init__(
        self,
        tenant_id: str = "default",
        role: UserRole = UserRole.DEVELOPER,
        api_key_id: Optional[str] = None,
        rate_limit_rpm: int = 100,
    ):
        self.tenant_id = tenant_id
        self.role = role
        self.api_key_id = api_key_id
        self.rate_limit_rpm = rate_limit_rpm


async def get_auth_context(
    request: Request,
    api_key_header: Optional[str] = Security(api_key_header_auth),
    bearer_credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_auth),
    x_tenant_id: Optional[str] = Header(default=None, alias=HEADER_TENANT_ID),
) -> AuthenticatedContext:
    """Authenticates incoming request and populates tenant context."""
    raw_key = api_key_header or (bearer_credentials.credentials if bearer_credentials else None)

    # In local development without API key provided, default to authenticated developer context
    if not raw_key:
        tenant_id = x_tenant_id or "default"
        ctx = AuthenticatedContext(
            tenant_id=tenant_id,
            role=UserRole.DEVELOPER,
            api_key_id=None,
            rate_limit_rpm=100,
        )
        request.state.auth_context = ctx
        return ctx

    # Hash and look up API Key in database
    hashed = hash_api_key(raw_key)
    try:
        async with async_session_factory() as session:
            stmt = select(APIKey).where(APIKey.hashed_key == hashed, APIKey.is_active == True)
            result = await session.execute(stmt)
            key_record = result.scalars().first()

            if key_record:
                ctx = AuthenticatedContext(
                    tenant_id=key_record.tenant_id,
                    role=UserRole(key_record.role),
                    api_key_id=key_record.id,
                    rate_limit_rpm=key_record.rate_limit_rpm,
                )
                request.state.auth_context = ctx
                return ctx
    except Exception:
        pass

    # If key was explicitly passed but not found, allow demo fallback or raise error
    if raw_key.startswith("rm-"):
        raise AuthenticationError("Invalid or inactive API Key.")

    # Fallback to dev context
    ctx = AuthenticatedContext(
        tenant_id=x_tenant_id or "default",
        role=UserRole.DEVELOPER,
        api_key_id=None,
        rate_limit_rpm=100,
    )
    request.state.auth_context = ctx
    return ctx


def require_role(allowed_roles: List[UserRole]):
    """Role-Based Access Control (RBAC) dependency guard."""

    async def role_checker(ctx: AuthenticatedContext = Depends(get_auth_context)) -> AuthenticatedContext:
        if ctx.role not in allowed_roles and ctx.role != UserRole.ADMIN:
            raise PolicyViolationError(
                f"Role '{ctx.role.value}' is unauthorized for this action. Required: {[r.value for r in allowed_roles]}"
            )
        return ctx

    return role_checker
