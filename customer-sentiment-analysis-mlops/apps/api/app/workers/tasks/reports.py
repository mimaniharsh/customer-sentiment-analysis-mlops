"""Background task: generate and upload executive reports to S3."""
from app.workers.celery_app import celery_app
import logging
logger = logging.getLogger(__name__)

@celery_app.task(name="app.workers.tasks.reports.generate_report", bind=True, max_retries=2)
def generate_report(self, report_id: str, tenant_id: str):
    logger.info("Generating report %s for tenant %s", report_id, tenant_id)
    # TODO: build PDF/XLSX report, upload to S3, update Report.status
    return {"status": "generated", "report_id": report_id}
