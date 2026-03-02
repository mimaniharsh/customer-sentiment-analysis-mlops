"""
apps/api/app/api/v1/endpoints/predict.py

Prediction endpoints with tenant isolation, input validation, and enriched ML output.
"""

from __future__ import annotations

import logging
from io import StringIO
from typing import Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field, field_validator

from app.ml.pipeline import get_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8192, description="Text to analyse")
    source: str = Field("api", description="Message source channel")
    external_id: Optional[str] = Field(None, description="Caller-side message ID")

    @field_validator("text")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=500)
    source: str = "api"

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        if any(len(t.strip()) == 0 for t in v):
            raise ValueError("Batch contains empty text entries")
        return [t.strip() for t in v]


def _format_result(result, text: str, idx: int = 0) -> dict:
    return {
        "id": idx,
        "text": text,
        "sentiment": result.sentiment,
        "sentiment_score": result.sentiment_score,
        "confidence": result.confidence,
        "probabilities": {
            "positive": result.prob_positive,
            "neutral": result.prob_neutral,
            "negative": result.prob_negative,
        },
        "urgency": result.urgency,
        "urgency_score": result.urgency_score,
        "severity_score": result.severity_score,
        "business_impact_score": result.business_impact_score,
        "attributes": result.attributes,
        "attribute_scores": result.attribute_scores,
        "explanation": result.explanation,
        "model_version": result.model_version,
        "latency_ms": result.latency_ms,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("", summary="Predict single text")
async def predict_single(request: Request, body: PredictRequest):
    """
    Analyse a single text and return sentiment, urgency, severity,
    business impact, multi-label attributes, and token-level explanation.
    """
    tenant_id = request.state.tenant_id
    logger.info("predict_single — tenant=%s  source=%s  len=%d", tenant_id, body.source, len(body.text))
    try:
        pipeline = get_pipeline()
        result = pipeline.predict(body.text)
        return _format_result(result, body.text)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="ML endpoint unavailable") from exc


@router.post("/batch", summary="Predict batch of texts")
async def predict_batch(request: Request, body: BatchPredictRequest):
    """
    Analyse up to 500 texts in a single call.
    Returns per-item results plus aggregate summary statistics.
    """
    tenant_id = request.state.tenant_id
    logger.info("predict_batch — tenant=%s  count=%d", tenant_id, len(body.texts))
    try:
        pipeline = get_pipeline()
        results = pipeline.predict_batch(body.texts)
        formatted = [_format_result(r, t, i) for i, (r, t) in enumerate(zip(results, body.texts))]

        pos = sum(1 for r in results if r.sentiment == "positive")
        neg = sum(1 for r in results if r.sentiment == "negative")
        neu = sum(1 for r in results if r.sentiment == "neutral")
        total = len(results)

        return {
            "total": total,
            "predictions": formatted,
            "summary": {
                "positive": pos,
                "neutral": neu,
                "negative": neg,
                "positive_rate": round(pos / total, 4) if total else 0,
                "negative_rate": round(neg / total, 4) if total else 0,
                "avg_confidence": round(sum(r.confidence for r in results) / total, 4) if total else 0,
                "avg_business_impact": round(sum(r.business_impact_score for r in results) / total, 4) if total else 0,
                "critical_count": sum(1 for r in results if r.urgency == "critical"),
            },
        }
    except Exception as exc:
        logger.error("Batch prediction failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="ML endpoint unavailable") from exc


@router.post("/csv", summary="Predict from CSV file")
async def predict_csv(request: Request, file: UploadFile = File(...)):
    """
    Upload a CSV file with a text column and receive sentiment analysis for every row.
    Accepted column names: text, tweet, message, content, review, comment, feedback.
    Max 10,000 rows.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    contents = await file.read()
    try:
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse CSV")

    if len(df) > 10_000:
        raise HTTPException(status_code=400, detail="CSV exceeds 10,000 row limit")

    text_col = next(
        (c for c in ["text", "tweet", "message", "content", "review", "comment", "feedback"] if c in df.columns),
        df.columns[0] if len(df.columns) >= 1 else None,
    )
    if text_col is None:
        raise HTTPException(status_code=400, detail="No recognisable text column found")

    texts = df[text_col].astype(str).tolist()
    pipeline = get_pipeline()
    results = pipeline.predict_batch(texts)
    formatted = [_format_result(r, t, i) for i, (r, t) in enumerate(zip(results, texts))]
    total = len(formatted)

    return {
        "filename": file.filename,
        "total": total,
        "predictions": formatted,
        "summary": {
            "positive": sum(1 for r in results if r.sentiment == "positive"),
            "neutral": sum(1 for r in results if r.sentiment == "neutral"),
            "negative": sum(1 for r in results if r.sentiment == "negative"),
            "avg_confidence": round(sum(r.confidence for r in results) / total, 4) if total else 0,
            "critical_count": sum(1 for r in results if r.urgency == "critical"),
        },
    }
