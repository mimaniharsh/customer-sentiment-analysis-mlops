"""apps/api/app/api/v1/endpoints/tenants.py"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

router = APIRouter()


class CreateTenantRequest(BaseModel):
    slug: str
    name: str
    plan: str = "starter"


@router.post("", status_code=status.HTTP_201_CREATED, summary="Create tenant")
async def create_tenant(body: CreateTenantRequest):
    # TODO: persist to DB
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="DB layer pending")


@router.get("/{tenant_slug}", summary="Get tenant details")
async def get_tenant(tenant_slug: str):
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="DB layer pending")
