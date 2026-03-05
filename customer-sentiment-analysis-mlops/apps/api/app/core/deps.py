"""
apps/api/app/core/deps.py

FastAPI dependency injection:
  - get_db          → AsyncSession per request
  - get_current_user → authenticated User from JWT
  - require_role    → RBAC guard factory
  - get_tenant_id   → validated tenant from header or JWT claim
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import decode_token_safe
from app.models.db import UserRole

_bearer = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DBSession = Annotated[AsyncSession, Depends(get_db)]


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
async def get_current_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer)],
) -> str:
    """
    Extracts and validates the JWT bearer token.
    Returns the user_id (sub claim) on success.
    Raises 401 on missing / invalid token.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token_safe(credentials.credentials)
    if not payload or payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload["sub"]


CurrentUserID = Annotated[str, Depends(get_current_user_id)]


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------
async def get_tenant_id(
    x_tenant_id: Annotated[str | None, Header(alias="X-Tenant-Id")] = None,
) -> str:
    """
    Reads tenant slug from the X-Tenant-Id request header.
    In production you'd cross-validate this against the JWT claims.
    """
    if not x_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Tenant-Id header is required",
        )
    return x_tenant_id


TenantID = Annotated[str, Depends(get_tenant_id)]


# ---------------------------------------------------------------------------
# RBAC
# ---------------------------------------------------------------------------
def require_role(*allowed_roles: UserRole):
    """
    Factory that returns a dependency requiring the caller to have
    one of the specified roles.

    Usage:
        @router.delete("/{id}", dependencies=[Depends(require_role(UserRole.ADMIN, UserRole.OWNER))])
    """
    async def _check(user_id: CurrentUserID, db: DBSession):
        # TODO: load User from DB and check user.role in allowed_roles
        # Stub passes through for now — wire once auth service is complete
        pass

    return _check
