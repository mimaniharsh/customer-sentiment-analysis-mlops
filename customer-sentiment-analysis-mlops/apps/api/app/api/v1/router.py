"""apps/api/app/api/v1/router.py"""
from fastapi import APIRouter

from app.api.v1.endpoints import predict, analytics, reports, auth, tenants, health

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(tenants.router, prefix="/tenants", tags=["Tenants"])
api_router.include_router(predict.router, prefix="/predict", tags=["Predictions"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
api_router.include_router(reports.router, prefix="/reports", tags=["Reports"])
api_router.include_router(health.router, prefix="/health", tags=["Health"])
