"""
apps/api/app/schemas/predict.py

Pydantic v2 schemas for prediction request/response contracts.
These are the public API shapes — separate from the DB models.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=8192)
    source: str = Field("api", description="Channel: tweet|review|ticket|chat|feedback|api")
    external_id: Optional[str] = None


class ProbabilitiesOut(BaseModel):
    positive: float
    neutral: float
    negative: float


class ExplainTokenOut(BaseModel):
    token: str
    importance: float


class ExplanationOut(BaseModel):
    sentiment_label: str
    top_tokens: list[ExplainTokenOut]
    method: str


class PredictOut(BaseModel):
    sentiment: str
    sentiment_score: float
    confidence: float
    probabilities: ProbabilitiesOut
    urgency: str
    urgency_score: float
    severity_score: float
    business_impact_score: float
    attributes: list[str]
    attribute_scores: dict[str, float]
    explanation: ExplanationOut
    model_version: str
    latency_ms: int


class BatchPredictIn(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=500)
    source: str = "api"


class BatchSummaryOut(BaseModel):
    positive: int
    neutral: int
    negative: int
    positive_rate: float
    negative_rate: float
    avg_confidence: float
    avg_business_impact: float
    critical_count: int


class BatchPredictOut(BaseModel):
    total: int
    predictions: list[dict]
    summary: BatchSummaryOut
