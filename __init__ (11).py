"""apps/api/app/api/v1/endpoints/health.py"""
from __future__ import annotations
import boto3
from fastapi import APIRouter
from app.core.config import get_settings

router = APIRouter()


@router.get("", summary="Deep health check")
async def deep_health():
    settings = get_settings()
    checks: dict = {}

    # SageMaker endpoint reachability
    try:
        sm = boto3.client("sagemaker", region_name=settings.AWS_REGION)
        resp = sm.describe_endpoint(EndpointName=settings.SAGEMAKER_ENDPOINT_NAME)
        checks["sagemaker"] = {
            "status": resp["EndpointStatus"],
            "healthy": resp["EndpointStatus"] == "InService",
        }
    except Exception as exc:
        checks["sagemaker"] = {"healthy": False, "error": str(exc)}

    all_healthy = all(v.get("healthy", False) for v in checks.values())
    return {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
    }
