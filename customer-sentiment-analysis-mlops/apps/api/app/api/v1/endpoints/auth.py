"""apps/api/app/api/v1/endpoints/auth.py"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr

router = APIRouter()


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


@router.post("/login", response_model=TokenResponse, summary="Obtain JWT token pair")
async def login(body: LoginRequest):
    """
    Authenticate with email + password and receive an access / refresh token pair.
    Tokens are short-lived JWTs signed with the configured JWT_SECRET_KEY.
    """
    # TODO: verify against DB, issue real JWT
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Auth DB layer not yet wired — implement in services/auth.py",
    )


@router.post("/refresh", summary="Refresh access token")
async def refresh_token(refresh_token: str):
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED)


@router.post("/logout", summary="Revoke tokens")
async def logout():
    return {"detail": "Logged out"}
