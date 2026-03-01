"""
apps/api/app/workers/celery_app.py

Celery application for background task processing.
- Metric snapshot computation (hourly)
- Drift detection (daily)
- Report generation
- Retraining workflow trigger
"""

from celery import Celery
from celery.schedules import crontab
from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "sentient",
    broker=settings.REDIS_CELERY_URL,
    backend=settings.REDIS_CELERY_URL,
    include=[
        "app.workers.tasks.metrics",
        "app.workers.tasks.drift",
        "app.workers.tasks.reports",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    beat_schedule={
        "compute-hourly-metrics": {
            "task": "app.workers.tasks.metrics.compute_metric_snapshots",
            "schedule": crontab(minute=0),   # every hour
        },
        "run-drift-detection": {
            "task": "app.workers.tasks.drift.detect_drift",
            "schedule": crontab(hour=2, minute=0),  # daily at 2 AM UTC
        },
    },
)
