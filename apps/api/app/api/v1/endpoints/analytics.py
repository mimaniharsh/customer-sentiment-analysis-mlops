"""
apps/api/app/api/v1/endpoints/analytics.py

Executive-grade analytics endpoints powering the dashboard.
Uses the in-memory prediction store for real data when available,
supplemented with deterministic mock historical data for the 30-day window.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Query, Request

from app.core.store import prediction_store

logger = logging.getLogger(__name__)
router = APIRouter()


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _mock_day(date_str: str) -> dict:
    """Generate one deterministic mock historical day."""
    rng = random.Random(hash(date_str) % 100_000)
    total = rng.randint(80, 300)
    pos = int(total * rng.uniform(0.35, 0.60))
    neg = int(total * rng.uniform(0.12, 0.28))
    neu = total - pos - neg
    return {
        "date": date_str,
        "total": total,
        "positive": pos,
        "neutral": neu,
        "negative": neg,
        "positive_rate": round(pos / total, 4),
        "negative_rate": round(neg / total, 4),
        "avg_sentiment_score": round(rng.uniform(-0.3, 0.7), 4),
        "avg_confidence": round(rng.uniform(0.72, 0.95), 4),
        "critical_count": rng.randint(0, 8),
        "avg_business_impact": round(rng.uniform(0.2, 0.7), 4),
    }


def _build_series(days: int) -> list[dict]:
    """
    Build a time-series for the past `days` days.
    Today uses real store data; history uses deterministic mock.
    """
    base = _now() - timedelta(days=days)
    today_str = _now().date().isoformat()
    series = []

    for i in range(days):
        d = base + timedelta(days=i)
        date_str = d.strftime("%Y-%m-%d")

        if date_str == today_str:
            records = prediction_store.today()
            if records:
                total = len(records)
                pos  = sum(1 for r in records if r["sentiment"] == "positive")
                neg  = sum(1 for r in records if r["sentiment"] == "negative")
                neu  = total - pos - neg
                crit = sum(1 for r in records if r["urgency"] == "critical")
                series.append({
                    "date": date_str,
                    "total": total,
                    "positive": pos,
                    "neutral": neu,
                    "negative": neg,
                    "positive_rate": round(pos / total, 4),
                    "negative_rate": round(neg / total, 4),
                    "avg_sentiment_score": round((pos - neg) / total, 4),
                    "avg_confidence": round(sum(r["confidence"] for r in records) / total, 4),
                    "critical_count": crit,
                    "avg_business_impact": round(sum(r["business_impact_score"] for r in records) / total, 4),
                })
                continue

        series.append(_mock_day(date_str))

    return series


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/overview", summary="Executive KPI overview")
async def overview(request: Request, days: int = Query(30, ge=1, le=365)):
    series = _build_series(days)
    total = sum(s["total"] for s in series)
    pos   = sum(s["positive"] for s in series)
    neg   = sum(s["negative"] for s in series)
    neu   = sum(s["neutral"]  for s in series)
    crit  = sum(s["critical_count"] for s in series)

    recent = series[-7:] if len(series) >= 7 else series
    prior  = series[-14:-7] if len(series) >= 14 else series[:7]
    r_neg  = sum(s["negative_rate"] for s in recent) / len(recent) if recent else 0
    p_neg  = sum(s["negative_rate"] for s in prior)  / len(prior)  if prior  else 0
    trend  = "improving" if r_neg < p_neg else "worsening" if r_neg > p_neg else "stable"

    return {
        "period_days": days,
        "live_today": len(prediction_store.today()),
        "kpis": {
            "total_messages": total,
            "positive_count": pos,
            "neutral_count": neu,
            "negative_count": neg,
            "positive_rate": round(pos / total, 4) if total else 0,
            "negative_rate": round(neg / total, 4) if total else 0,
            "critical_urgency_count": crit,
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
    return {"granularity": granularity, "series": _build_series(days)}


@router.get("/live", summary="Live session stats from current predictions")
async def live_stats(request: Request):
    """Real-time stats from all predictions made since server start."""
    records = prediction_store.snapshot()
    total = len(records)
    if not total:
        return {"total": 0, "message": "No predictions yet — run the Live Stream demo!"}

    pos  = sum(1 for r in records if r["sentiment"] == "positive")
    neg  = sum(1 for r in records if r["sentiment"] == "negative")
    neu  = total - pos - neg
    crit = sum(1 for r in records if r["urgency"] == "critical")
    high = sum(1 for r in records if r["urgency"] == "high")
    avg_conf   = sum(r["confidence"]           for r in records) / total
    avg_impact = sum(r["business_impact_score"] for r in records) / total

    attr_counts: dict[str, int] = defaultdict(int)
    for r in records:
        for a in r.get("attributes", []):
            attr_counts[a] += 1
    top_attrs = sorted(attr_counts.items(), key=lambda x: -x[1])[:5]

    return {
        "total": total,
        "positive": pos,
        "neutral": neu,
        "negative": neg,
        "critical": crit,
        "high_urgency": high,
        "positive_rate": round(pos / total, 4),
        "negative_rate": round(neg / total, 4),
        "avg_confidence": round(avg_conf, 4),
        "avg_business_impact": round(avg_impact, 4),
        "top_attributes": [{"attribute": a, "count": c} for a, c in top_attrs],
    }


@router.get("/attributes", summary="Attribute heatmap data")
async def attributes(request: Request, days: int = Query(30, ge=1, le=90)):
    records = prediction_store.snapshot()
    attrs_list = ["pricing", "support", "reliability", "onboarding", "features", "performance", "security"]
    sentiments = ["positive", "neutral", "negative"]
    real: dict[tuple, list] = defaultdict(list)
    for r in records:
        for a in r.get("attributes", []):
            if a in attrs_list:
                real[(a, r["sentiment"])].append(r["business_impact_score"])

    heatmap = []
    rng = random.Random(99)
    for attr in attrs_list:
        for sentiment in sentiments:
            key = (attr, sentiment)
            if key in real and real[key]:
                count = len(real[key])
                avg_score = sum(real[key]) / count
            else:
                count = rng.randint(5, 120)
                avg_score = rng.uniform(0.1, 0.9)
            heatmap.append({
                "attribute": attr,
                "sentiment": sentiment,
                "count": count,
                "avg_score": round(avg_score, 3),
            })
    return {"period_days": days, "heatmap": heatmap}


@router.get("/urgency", summary="Urgency & severity trends")
async def urgency_trends(request: Request, days: int = Query(30, ge=1, le=90)):
    today_str = _now().date().isoformat()
    records_today = prediction_store.today()
    rng = random.Random(77)
    series = []
    for i in range(days):
        date = (_now() - timedelta(days=days - i)).strftime("%Y-%m-%d")
        if date == today_str and records_today:
            series.append({
                "date": date,
                "critical": sum(1 for r in records_today if r["urgency"] == "critical"),
                "high":     sum(1 for r in records_today if r["urgency"] == "high"),
                "medium":   sum(1 for r in records_today if r["urgency"] == "medium"),
                "low":      sum(1 for r in records_today if r["urgency"] == "low"),
            })
        else:
            series.append({
                "date": date,
                "critical": rng.randint(0, 5),
                "high":     rng.randint(3, 15),
                "medium":   rng.randint(10, 40),
                "low":      rng.randint(20, 80),
            })
    return {"series": series}


@router.get("/drift", summary="Model drift indicators")
async def drift(request: Request):
    rng = random.Random(11)
    records = []
    for i in range(12):
        week_start = (_now() - timedelta(weeks=12 - i)).strftime("%Y-%m-%d")
        psi = round(rng.uniform(0.02, 0.35), 4)
        records.append({
            "week_start": week_start,
            "psi_score": psi,
            "ks_statistic": round(rng.uniform(0.03, 0.2), 4),
            "drift_detected": psi > 0.2,
            "alert": psi > 0.25,
        })
    return {"drift_history": records, "current_psi": records[-1]["psi_score"]}


@router.get("/clusters", summary="Root-cause sentiment clusters")
async def clusters(request: Request):
    records = prediction_store.snapshot()
    rng = random.Random(55)
    themes = [
        ("Billing & Refunds",   "negative"),
        ("Feature Requests",    "neutral"),
        ("Service Outages",     "negative"),
        ("Onboarding Praise",   "positive"),
        ("Support Quality",     "negative"),
        ("Product Performance", "positive"),
        ("Security Concerns",   "negative"),
        ("UX Improvements",     "neutral"),
    ]
    attr_map = {"pricing": 0, "support": 4, "reliability": 2, "features": 1, "performance": 5, "security": 6, "onboarding": 3}
    real_counts: dict[int, int] = defaultdict(int)
    for r in records:
        for a in r.get("attributes", []):
            if a in attr_map:
                real_counts[attr_map[a]] += 1

    cluster_data = []
    for idx, (theme, dominant) in enumerate(themes):
        count = real_counts.get(idx) or rng.randint(20, 200)
        cluster_data.append({
            "cluster_id": idx,
            "theme": theme,
            "dominant_sentiment": dominant,
            "message_count": count,
            "avg_severity": round(rng.uniform(0.2, 0.9), 3),
            "avg_business_impact": round(rng.uniform(0.15, 0.85), 3),
            "top_terms": rng.sample(
                ["refund", "broken", "slow", "amazing", "confusing", "expensive", "helpful", "crash"], k=4,
            ),
        })
    return {"clusters": cluster_data}


@router.get("/business-impact", summary="Business impact simulation")
async def business_impact(
    request: Request,
    resolve_critical_pct: float = Query(0.5, ge=0.0, le=1.0),
    resolve_high_pct:     float = Query(0.3, ge=0.0, le=1.0),
):
    records = prediction_store.snapshot()
    total = len(records)
    if total:
        neg = sum(1 for r in records if r["sentiment"] == "negative")
        base_score = round((1 - neg / total) * 100, 1)
    else:
        base_score = 62.5

    simulated = base_score + (resolve_critical_pct * 15) + (resolve_high_pct * 8)
    return {
        "baseline_health_score": base_score,
        "simulated_health_score": round(min(simulated, 100), 1),
        "improvement": round(simulated - base_score, 1),
        "inputs": {"resolve_critical_pct": resolve_critical_pct, "resolve_high_pct": resolve_high_pct},
    }
