"""
apps/api/app/services/prediction_service.py

Business logic layer for persisting and querying predictions.
Sits between the API endpoints and the database models.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.pipeline import PipelineResult
from app.models.db import Message, MessageSource, Prediction, SentimentLabel, UrgencyLevel

logger = logging.getLogger(__name__)


async def save_prediction(
    db: AsyncSession,
    tenant_id: str,
    text: str,
    result: PipelineResult,
    source: str = "api",
    external_id: Optional[str] = None,
) -> Prediction:
    """Persist a message + its prediction to the database."""

    # 1. Save the raw message
    message = Message(
        tenant_id=tenant_id,
        raw_text=text,
        source=MessageSource(source) if source in [e.value for e in MessageSource] else MessageSource.API,
        external_id=external_id,
        language="en",
    )
    db.add(message)
    await db.flush()  # get message.id without committing

    # 2. Save the enriched prediction
    prediction = Prediction(
        tenant_id=tenant_id,
        message_id=message.id,
        sentiment=SentimentLabel(result.sentiment),
        sentiment_score=result.sentiment_score,
        confidence=result.confidence,
        prob_positive=result.prob_positive,
        prob_neutral=result.prob_neutral,
        prob_negative=result.prob_negative,
        urgency=UrgencyLevel(result.urgency),
        urgency_score=result.urgency_score,
        severity_score=result.severity_score,
        business_impact_score=result.business_impact_score,
        attributes=result.attributes,
        attribute_scores=result.attribute_scores,
        explanation=result.explanation,
        model_version=result.model_version,
        endpoint_name=result.endpoint_name,
        latency_ms=result.latency_ms,
        embedding_id=result.embedding_id,
    )
    db.add(prediction)
    await db.flush()

    logger.debug(
        "Saved prediction id=%s tenant=%s sentiment=%s confidence=%.3f",
        prediction.id,
        tenant_id,
        result.sentiment,
        result.confidence,
    )
    return prediction


async def get_recent_predictions(
    db: AsyncSession,
    tenant_id: str,
    limit: int = 100,
    offset: int = 0,
) -> list[Prediction]:
    """Fetch recent predictions for a tenant, newest first."""
    result = await db.execute(
        select(Prediction)
        .where(Prediction.tenant_id == tenant_id)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().all())


async def get_sentiment_counts(
    db: AsyncSession,
    tenant_id: str,
    since: Optional[datetime] = None,
) -> dict[str, int]:
    """Aggregate sentiment label counts for a tenant over a time window."""
    query = select(
        Prediction.sentiment,
        func.count(Prediction.id).label("count"),
    ).where(Prediction.tenant_id == tenant_id)

    if since:
        query = query.where(Prediction.created_at >= since)

    query = query.group_by(Prediction.sentiment)
    rows = await db.execute(query)
    return {row.sentiment.value: row.count for row in rows}
