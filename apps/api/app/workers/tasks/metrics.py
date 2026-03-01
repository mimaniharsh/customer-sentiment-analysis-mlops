"""Background task: compute hourly metric snapshots per tenant."""
from app.workers.celery_app import celery_app
import logging
logger = logging.getLogger(__name__)

@celery_app.task(name="app.workers.tasks.metrics.compute_metric_snapshots", bind=True, max_retries=3)
def compute_metric_snapshots(self):
    """Aggregate predictions into MetricSnapshot rows for dashboard efficiency."""
    logger.info("Computing metric snapshots...")
    # TODO: implement DB aggregation query
    return {"status": "ok"}
