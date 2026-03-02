"""
apps/api/app/api/v1/endpoints/analytics.py

Executive-grade analytics endpoints powering the dashboard.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)
router = APIRouter()


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _mock_time_series(days: int = 30) -> list[dict]:
    """Generate mock KPI time-series until DB layer is wired."""
    import random
    random.seed(42)
    base = _now() - timedelta(days=days)
    series = []
    for i in range(days):
        d = base + timedelta(days=i)
        total = random.randint(80, 300)
        pos = int(total * random.uniform(0.35, 0.65))
        neg = int(total * random.uniform(0.10, 0.30))
        neu = total - pos - neg
        series.append({
            "date": d.strftime("%Y-%m-%d"),
            "total": total,
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "positive_rate": round(pos / total, 4),
            "negative_rate": round(neg / total, 4),
            "avg_sentiment_score": round(random.uniform(-0.3, 0.7), 4),
            "avg_confidence": round(random.uniform(0.72, 0.95), 4),
            "critical_count": random.randint(0, 8),
            "avg_business_impact": round(random.uniform(0.2, 0.7), 4),
        })
    return series


@router.get("/overview", summary="Executive KPI overview")
async def overview(request: Request, days: int = Query(30, ge=1, le=365)):
    """
    Top-level KPIs for the executive dashboard:
    sentiment distribution, trend direction, business impact score,
    urgency breakdown, and confidence health.
    """
    tenant_id = request.state.tenant_id
    series = _mock_time_series(days)

    total = sum(s["total"] for s in series)
    pos = sum(s["positive"] for s in series)
    neg = sum(s["negative"] for s in series)
    neu = sum(s["neutral"] for s in series)
    critical = sum(s["critical_count"] for s in series)

    # Trend: compare last 7 days vs prior 7 days
    recent = series[-7:] if len(series) >= 7 else series
    prior = series[-14:-7] if len(series) >= 14 else series[:7]
    recent_neg_rate = sum(s["negative_rate"] for s in recent) / len(recent) if recent else 0
    prior_neg_rate = sum(s["negative_rate"] for s in prior) / len(prior) if prior else 0
    trend = "improving" if recent_neg_rate < prior_neg_rate else "worsening" if recent_neg_rate > prior_neg_rate else "stable"

    return {
        "period_days": days,
        "kpis": {
            "total_messages": total,
            "positive_count": pos,
            "neutral_count": neu,
            "negative_count": neg,
            "positive_rate": round(pos / total, 4) if total else 0,
            "negative_rate": round(neg / total, 4) if total else 0,
            "critical_urgency_count": critical,
            "avg_business_impact": round(
                sum(s["avg_business_impact"] for s in series) / len(series), 4
            ) if series else 0,
            "avg_confidence": round(
                sum(s["avg_confidence"] for s in series) / len(series), 4
            ) if series else 0,
        },
        "trend": trend,
        "sentiment_health_score": round((pos / total * 100) if total else 50, 1),
    }


@router.get("/timeseries", summary="Sentiment time-series")
async def timeseries(
    request: Request,
    days: int = Query(30, ge=1, le=365),
    granularity: str = Query("day", pattern="^(hour|day|week|month)$"),
):
    """Daily/weekly/monthly sentiment trend data for charts."""
    return {"granularity": granularity, "series": _mock_time_series(days)}


@router.get("/attributes", summary="Attribute heatmap data")
async def attributes(request: Request, days: int = Query(30, ge=1, le=90)):
    """Returns attribute distribution scores for the heatmap widget."""
    import random
    random.seed(99)
    attrs = ["pricing", "support", "reliability", "onboarding", "features", "performance", "security"]
    sentiments = ["positive", "neutral", "negative"]
    heatmap = []
    for attr in attrs:
        for sentiment in sentiments:
            heatmap.append({
                "attribute": attr,
                "sentiment": sentiment,
                "count": random.randint(5, 120),
                "avg_score": round(random.uniform(0.1, 0.9), 3),
            })
    return {"period_days": days, "heatmap": heatmap}


@router.get("/urgency", summary="Urgency & severity trends")
async def urgency_trends(request: Request, days: int = Query(30, ge=1, le=90)):
    """Urgency level breakdown over time."""
    import random
    random.seed(77)
    series = []
    for i in range(days):
        date = (_now() - timedelta(days=days - i)).strftime("%Y-%m-%d")
        series.append({
            "date": date,
            "critical": random.randint(0, 5),
            "high": random.randint(3, 15),
            "medium": random.randint(10, 40),
            "low": random.randint(20, 80),
        })
    return {"series": series}


@router.get("/drift", summary="Model drift indicators")
async def drift(request: Request):
    """PSI and KS drift statistics for model monitoring panel."""
    import random
    random.seed(11)
    records = []
    for i in range(12):
        week_start = (_now() - timedelta(weeks=12 - i)).strftime("%Y-%m-%d")
        psi = round(random.uniform(0.02, 0.35), 4)
        records.append({
            "week_start": week_start,
            "psi_score": psi,
            "ks_statistic": round(random.uniform(0.03, 0.2), 4),
            "drift_detected": psi > 0.2,
            "alert": psi > 0.25,
        })
    return {"drift_history": records, "current_psi": records[-1]["psi_score"]}


@router.get("/clusters", summary="Root-cause sentiment clusters")
async def clusters(request: Request):
    """Embedding-based cluster summaries for root-cause analysis."""
    import random
    random.seed(55)
    cluster_data = []
    themes = [
        ("Billing & Refunds", "negative"),
        ("Feature Requests", "neutral"),
        ("Service Outages", "negative"),
        ("Onboarding Praise", "positive"),
        ("Support Quality", "negative"),
        ("Product Performance", "positive"),
        ("Security Concerns", "negative"),
        ("UX Improvements", "neutral"),
    ]
    for idx, (theme, dominant_sentiment) in enumerate(themes):
        cluster_data.append({
            "cluster_id": idx,
            "theme": theme,
            "dominant_sentiment": dominant_sentiment,
            "message_count": random.randint(20, 200),
            "avg_severity": round(random.uniform(0.2, 0.9), 3),
            "avg_business_impact": round(random.uniform(0.15, 0.85), 3),
            "top_terms": random.sample(
                ["refund", "broken", "slow", "amazing", "confusing", "expensive", "helpful", "crash"],
                k=4,
            ),
        })
    return {"clusters": cluster_data}


@router.get("/business-impact", summary="Business impact simulation")
async def business_impact(
    request: Request,
    resolve_critical_pct: float = Query(0.5, ge=0.0, le=1.0),
    resolve_high_pct: float = Query(0.3, ge=0.0, le=1.0),
):
    """
    Scenario simulation: what happens to overall sentiment health score
    if N% of critical/high urgency items are resolved?
    """
    base_score = 62.5
    simulated = base_score + (resolve_critical_pct * 15) + (resolve_high_pct * 8)
    return {
        "baseline_health_score": base_score,
        "simulated_health_score": round(min(simulated, 100), 1),
        "improvement": round(simulated - base_score, 1),
        "inputs": {
            "resolve_critical_pct": resolve_critical_pct,
            "resolve_high_pct": resolve_high_pct,
        },
    }
