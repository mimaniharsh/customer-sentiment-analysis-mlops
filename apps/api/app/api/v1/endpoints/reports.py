"""apps/api/app/api/v1/endpoints/reports.py"""
from fastapi import APIRouter
router = APIRouter()

@router.get("", summary="List reports")
async def list_reports():
    return {"reports": []}

@router.post("", summary="Generate report")
async def generate_report():
    return {"status": "queued", "report_id": "placeholder"}
