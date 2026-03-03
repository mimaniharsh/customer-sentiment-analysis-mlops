"""
apps/api/tests/test_predict.py

Integration-level tests for the prediction and analytics endpoints.
The SageMaker pipeline is mocked — no real AWS calls are made in tests.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestPredictEndpoint:
    def test_single_predict_returns_all_fields(self, client: TestClient):
        resp = client.post(
            "/api/v1/predict",
            json={"text": "The product is absolutely fantastic, exceeded all expectations!"},
            headers={"X-Tenant-Id": "test-tenant"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["sentiment"] in ("positive", "neutral", "negative")
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["urgency"] in ("critical", "high", "medium", "low")
        assert "attributes" in data
        assert "explanation" in data
        assert "business_impact_score" in data
        assert "latency_ms" in data

    def test_empty_text_is_rejected(self, client: TestClient):
        resp = client.post(
            "/api/v1/predict",
            json={"text": ""},
            headers={"X-Tenant-Id": "test-tenant"},
        )
        assert resp.status_code == 422

    def test_text_too_long_is_rejected(self, client: TestClient):
        resp = client.post(
            "/api/v1/predict",
            json={"text": "x" * 9000},
            headers={"X-Tenant-Id": "test-tenant"},
        )
        assert resp.status_code == 422

    def test_batch_predict_returns_summary(self, client: TestClient):
        resp = client.post(
            "/api/v1/predict/batch",
            json={"texts": ["Great service!", "Terrible experience", "It was okay"]},
            headers={"X-Tenant-Id": "test-tenant"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert len(data["predictions"]) == 3
        assert "summary" in data
        assert "positive" in data["summary"]
        assert "avg_confidence" in data["summary"]

    def test_batch_empty_is_rejected(self, client: TestClient):
        resp = client.post(
            "/api/v1/predict/batch",
            json={"texts": []},
            headers={"X-Tenant-Id": "test-tenant"},
        )
        assert resp.status_code == 422


class TestHealthEndpoint:
    def test_liveness_probe(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_version_in_response(self, client: TestClient):
        resp = client.get("/health")
        assert "version" in resp.json()


class TestAnalyticsEndpoint:
    def test_overview_default_period(self, client: TestClient):
        resp = client.get(
            "/api/v1/analytics/overview",
            headers={"X-Tenant-Id": "test-tenant"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "kpis" in data
        assert "trend" in data
        assert "sentiment_health_score" in data

    def test_timeseries_returns_series(self, client: TestClient):
        resp = client.get(
            "/api/v1/analytics/timeseries?days=7",
            headers={"X-Tenant-Id": "test-tenant"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "series" in data
        assert len(data["series"]) == 7

    def test_drift_returns_history(self, client: TestClient):
        resp = client.get(
            "/api/v1/analytics/drift",
            headers={"X-Tenant-Id": "test-tenant"},
        )
        assert resp.status_code == 200
        assert "drift_history" in resp.json()

    def test_clusters_endpoint(self, client: TestClient):
        resp = client.get(
            "/api/v1/analytics/clusters",
            headers={"X-Tenant-Id": "test-tenant"},
        )
        assert resp.status_code == 200
        clusters = resp.json()["clusters"]
        assert len(clusters) > 0
        assert "theme" in clusters[0]
