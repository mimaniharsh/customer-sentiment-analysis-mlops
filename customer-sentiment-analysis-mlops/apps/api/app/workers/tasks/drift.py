"""Background task: run drift detection across active tenants."""
from app.workers.celery_app import celery_app
from app.ml.pipeline import DriftDetector
import logging
logger = logging.getLogger(__name__)

@celery_app.task(name="app.workers.tasks.drift.detect_drift", bind=True, max_retries=2)
def detect_drift(self):
    """Compute PSI and KS drift statistics and store DriftRecord."""
    logger.info("Running drift detection...")
    # TODO: fetch recent scores from DB, call DriftDetector.check_drift()
    return {"status": "ok"}
