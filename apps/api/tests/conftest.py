"""
apps/api/tests/conftest.py

Pytest fixtures for the Sentient API test suite.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture(scope="session")
def mock_settings(monkeypatch=None):
    """Patch settings so tests don't require real AWS credentials."""
    return {
        "APP_ENV": "test",
        "APP_SECRET_KEY": "test-secret-key-32-chars-long-here",
        "JWT_SECRET_KEY": "test-jwt-key-32-chars-long-here!!",
        "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
        "REDIS_URL": "redis://localhost:6379/9",
        "REDIS_CELERY_URL": "redis://localhost:6379/9",
        "AWS_REGION": "ap-south-1",
        "SAGEMAKER_ENDPOINT_NAME": "test-endpoint",
        "S3_BUCKET": "test-bucket",
    }


@pytest.fixture
def mock_pipeline_result():
    """A realistic PipelineResult for mocking the ML pipeline."""
    from app.ml.pipeline import PipelineResult
    return PipelineResult(
        sentiment="positive",
        sentiment_score=0.72,
        confidence=0.88,
        prob_positive=0.82,
        prob_neutral=0.12,
        prob_negative=0.06,
        urgency="low",
        urgency_score=0.12,
        severity_score=0.09,
        business_impact_score=0.14,
        attributes=["support"],
        attribute_scores={"support": 0.45},
        explanation={
            "sentiment_label": "positive",
            "top_tokens": [{"token": "great", "importance": 0.42}],
            "method": "attention_proxy",
        },
        model_version="v1",
        endpoint_name="test-endpoint",
        latency_ms=87,
        embedding_id="abc123",
    )


@pytest.fixture
def client(mock_pipeline_result):
    """FastAPI test client with ML pipeline mocked out."""
    with patch("app.ml.pipeline.get_pipeline") as mock_get:
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = mock_pipeline_result
        mock_pipeline.predict_batch.return_value = [mock_pipeline_result]
        mock_get.return_value = mock_pipeline

        # Patch config validation so tests don't need real keys
        with patch("app.core.config.Settings.validate_required"):
            from app.main import app
            yield TestClient(app)
