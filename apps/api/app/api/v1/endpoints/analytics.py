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


@router.get("/bubble", summary="Bubble chart data — attribute × severity × volume")
async def bubble_chart(request: Request):
    """
    Aggregates live prediction data into bubble chart format.

    Returns one bubble per attribute with:
      - message_count   → bubble size
      - avg_severity    → Y axis
      - negative_rate   → colour intensity
      - top_issues      → root cause breakdown per attribute
      - issue_breakdown → full issue × count list for drill-down
    """
    from app.ml.pipeline import ISSUE_TAXONOMY

    records = prediction_store.snapshot()
    rng = random.Random(77)

    # Attributes we care about
    attrs = ["reliability", "support", "pricing", "performance", "onboarding", "features", "security"]

    # Aggregate real data from prediction store
    attr_data: dict[str, dict] = {
        a: {
            "count": 0,
            "neg_count": 0,
            "severity_sum": 0.0,
            "impact_sum": 0.0,
            "issue_counts": defaultdict(int),
        }
        for a in attrs
    }

    for r in records:
        for a in r.get("attributes", []):
            if a not in attr_data:
                continue
            d = attr_data[a]
            d["count"] += 1
            if r.get("sentiment") == "negative":
                d["neg_count"] += 1
            d["severity_sum"] += r.get("severity_score", 0.0)
            d["impact_sum"] += r.get("business_impact_score", 0.0)
            # Count issues from theme_breakdown
            for tb in r.get("theme_breakdown", []):
                if tb.get("attribute") == a:
                    issue = tb.get("issue", "General feedback")
                    d["issue_counts"][issue] += 1

    # Build output; fall back to seeded random for empty attributes
    ATTR_BASELINE = {
        "reliability": (85, 0.78, 0.82),
        "support":     (62, 0.65, 0.71),
        "pricing":     (44, 0.58, 0.63),
        "performance": (38, 0.41, 0.47),
        "onboarding":  (22, 0.33, 0.38),
        "features":    (18, 0.24, 0.29),
        "security":    (11, 0.51, 0.56),
    }

    bubbles = []
    for attr in attrs:
        d = attr_data[attr]
        base_count, base_sev, base_neg_rate = ATTR_BASELINE[attr]

        count = d["count"] if d["count"] > 0 else base_count + rng.randint(-8, 8)
        avg_severity = round(d["severity_sum"] / d["count"], 3) if d["count"] > 0 else round(base_sev + rng.uniform(-0.05, 0.05), 3)
        neg_rate = round(d["neg_count"] / d["count"], 3) if d["count"] > 0 else round(base_neg_rate + rng.uniform(-0.04, 0.04), 3)
        avg_impact = round(d["impact_sum"] / d["count"], 3) if d["count"] > 0 else round(rng.uniform(0.3, 0.8), 3)

        # Issue breakdown: use real data or fallback to taxonomy sample
        if d["issue_counts"]:
            issue_breakdown = [
                {"issue": iss, "count": cnt}
                for iss, cnt in sorted(d["issue_counts"].items(), key=lambda x: -x[1])
            ]
        else:
            # Generate plausible fallback from taxonomy
            if attr in ISSUE_TAXONOMY:
                issues = list(ISSUE_TAXONOMY[attr].keys())
                issue_breakdown = [
                    {"issue": iss, "count": rng.randint(3, base_count // max(len(issues), 1) + 5)}
                    for iss in issues
                ]
                issue_breakdown.sort(key=lambda x: -x["count"])
            else:
                issue_breakdown = [{"issue": "General feedback", "count": count}]

        top_issues = [ib["issue"] for ib in issue_breakdown[:3]]

        # Risk tier derived from avg_severity
        if avg_severity >= 0.75:
            risk_tier = "CRITICAL"
        elif avg_severity >= 0.50:
            risk_tier = "HIGH"
        elif avg_severity >= 0.25:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "LOW"

        bubbles.append({
            "attribute":      attr,
            "message_count":  count,
            "avg_severity":   avg_severity,
            "avg_impact":     avg_impact,
            "negative_rate":  neg_rate,
            "risk_tier":      risk_tier,
            "top_issues":     top_issues,
            "issue_breakdown": issue_breakdown,
        })

    # Sort by message_count descending (highest volume first)
    bubbles.sort(key=lambda b: -b["message_count"])

    return {
        "bubbles": bubbles,
        "total_messages": sum(b["message_count"] for b in bubbles),
        "highest_risk_attribute": max(bubbles, key=lambda b: b["avg_severity"])["attribute"],
        "highest_volume_attribute": bubbles[0]["attribute"] if bubbles else None,
    }


@router.get("/issues", summary="Ranked issue pinpointer — exact root causes across all attributes")
async def issues_pinpointer(request: Request):
    """
    Flattens all issue_breakdown data from the bubble endpoint into a
    single ranked list of specific root-cause issues.

    Each entry includes:
      - issue            : specific problem name (e.g. "Flight delay")
      - attribute        : parent category (e.g. "reliability")
      - count            : number of mentions
      - priority_score   : count × parent avg_severity (composite risk weight)
      - risk_tier        : CRITICAL / HIGH / MEDIUM / LOW
      - negative_rate    : proportion of negative signals for parent attribute
    """
    from app.ml.pipeline import ISSUE_TAXONOMY

    records = prediction_store.snapshot()
    rng = random.Random(91)

    attrs = ["reliability", "support", "pricing", "performance", "onboarding", "features", "security"]

    # --- Aggregate real data ---
    attr_meta: dict[str, dict] = {
        a: {"count": 0, "neg_count": 0, "severity_sum": 0.0, "issue_counts": defaultdict(int)}
        for a in attrs
    }
    for r in records:
        for a in r.get("attributes", []):
            if a not in attr_meta:
                continue
            m = attr_meta[a]
            m["count"] += 1
            if r.get("sentiment") == "negative":
                m["neg_count"] += 1
            m["severity_sum"] += r.get("severity_score", 0.0)
            for tb in r.get("theme_breakdown", []):
                if tb.get("attribute") == a:
                    m["issue_counts"][tb.get("issue", "General feedback")] += 1

    # --- Baseline severities per attribute (matches bubble endpoint) ---
    ATTR_BASELINE = {
        "reliability": (85, 0.78, 0.82),
        "support":     (62, 0.65, 0.71),
        "pricing":     (44, 0.58, 0.63),
        "performance": (38, 0.41, 0.47),
        "onboarding":  (22, 0.33, 0.38),
        "features":    (18, 0.24, 0.29),
        "security":    (11, 0.51, 0.56),
    }

    flat_issues = []

    for attr in attrs:
        m = attr_meta[attr]
        base_count, base_sev, base_neg = ATTR_BASELINE[attr]

        attr_count = m["count"] if m["count"] > 0 else base_count
        avg_sev = round(m["severity_sum"] / m["count"], 3) if m["count"] > 0 else round(base_sev + rng.uniform(-0.04, 0.04), 3)
        neg_rate = round(m["neg_count"] / m["count"], 3) if m["count"] > 0 else round(base_neg + rng.uniform(-0.03, 0.03), 3)

        # Risk tier from avg_severity
        if avg_sev >= 0.75:
            risk_tier = "CRITICAL"
        elif avg_sev >= 0.50:
            risk_tier = "HIGH"
        elif avg_sev >= 0.25:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "LOW"

        # Build issue list for this attribute
        if m["issue_counts"]:
            issue_pairs = list(m["issue_counts"].items())
        elif attr in ISSUE_TAXONOMY:
            issues = list(ISSUE_TAXONOMY[attr].keys())
            issue_pairs = [
                (iss, rng.randint(3, max(attr_count // max(len(issues), 1), 4)))
                for iss in issues
            ]
        else:
            issue_pairs = [("General feedback", attr_count)]

        for issue_name, cnt in issue_pairs:
            priority_score = round(cnt * avg_sev, 3)
            flat_issues.append({
                "issue":          issue_name,
                "attribute":      attr,
                "count":          cnt,
                "priority_score": priority_score,
                "avg_severity":   avg_sev,
                "negative_rate":  neg_rate,
                "risk_tier":      risk_tier,
            })

    # Sort by priority_score descending
    flat_issues.sort(key=lambda x: -x["priority_score"])

    # Normalise count for bar widths (0–1 relative to max)
    max_count = max((i["count"] for i in flat_issues), default=1)
    for i in flat_issues:
        i["count_pct"] = round(i["count"] / max_count, 4)

    return {
        "issues":       flat_issues,
        "total_issues": len(flat_issues),
        "top_issue":    flat_issues[0]["issue"] if flat_issues else None,
        "top_attribute": flat_issues[0]["attribute"] if flat_issues else None,
    }


@router.get("/strategy", summary="MBA-grade strategic intelligence — brand health, revenue risk, priority matrix, action plan")
async def strategy_intelligence(request: Request):
    """
    Returns a board-ready strategic intelligence package:
      - Brand health scorecard (per attribute, RAG status, trend, churn risk)
      - Revenue at risk model (customer funnel → $ exposure)
      - Strategic priority matrix (impact × effort quadrants)
      - Executive action plan (ranked, owned, time-boxed)
    """
    records = prediction_store.snapshot()
    rng = random.Random(42)

    ATTRS = ["reliability", "support", "pricing", "performance", "onboarding", "features", "security"]

    # Real data aggregation
    attr_data: dict[str, dict] = {
        a: {"count": 0, "neg": 0, "sev_sum": 0.0, "impact_sum": 0.0}
        for a in ATTRS
    }
    for r in records:
        for a in r.get("attributes", []):
            if a in attr_data:
                d = attr_data[a]
                d["count"] += 1
                if r.get("sentiment") == "negative":
                    d["neg"] += 1
                d["sev_sum"]    += r.get("severity_score", 0.0)
                d["impact_sum"] += r.get("business_impact_score", 0.0)

    # Attribute config: base_score, base_neg_rate, base_churn_risk, avg_clv_exposure, effort_to_fix
    ATTR_CFG = {
        "reliability": {"score": 34, "neg": 0.72, "churn": 0.68, "clv": 3200, "effort": 0.75},
        "support":     {"score": 52, "neg": 0.58, "churn": 0.51, "clv": 2100, "effort": 0.35},
        "pricing":     {"score": 41, "neg": 0.65, "churn": 0.61, "clv": 4500, "effort": 0.55},
        "performance": {"score": 63, "neg": 0.38, "churn": 0.33, "clv": 1800, "effort": 0.45},
        "onboarding":  {"score": 71, "neg": 0.25, "churn": 0.21, "clv": 900,  "effort": 0.25},
        "features":    {"score": 68, "neg": 0.22, "churn": 0.18, "clv": 700,  "effort": 0.60},
        "security":    {"score": 78, "neg": 0.14, "churn": 0.09, "clv": 5200, "effort": 0.85},
    }

    RECOMMENDATIONS = {
        "reliability": "Immediate ops escalation. Deploy real-time SLA monitoring & auto-paging.",
        "support":     "Staff augmentation + AI triage chatbot. Target < 4h first-response SLA.",
        "pricing":     "Transparent billing redesign. Introduce tier-lock pricing guarantee.",
        "performance": "CDN + caching rollout. Target p95 latency under 200ms.",
        "onboarding":  "Guided tour redesign with milestone email nudges at days 1 / 7 / 30.",
        "features":    "Roadmap transparency. Ship top-3 voted features in next sprint cycle.",
        "security":    "Maintain SOC-2 posture. Publish pen-test results on trust page.",
    }

    ATTR_COLOUR = {
        "reliability": "#f97316", "support": "#8b5cf6", "pricing": "#ef4444",
        "performance": "#3b82f6", "onboarding": "#10b981", "features": "#f59e0b",
        "security":    "#ec4899",
    }

    # Priority matrix issues per attribute: (issue_name, impact 0-1, effort 0-1)
    ATTR_ISSUES = {
        "reliability": [("Service outages",      0.88, 0.75), ("Delivery delays",      0.82, 0.65), ("Data sync failures",   0.71, 0.80)],
        "support":     [("Long hold times",       0.74, 0.30), ("Unhelpful agents",     0.68, 0.40), ("No follow-up",         0.55, 0.20)],
        "pricing":     [("Hidden fees",           0.78, 0.45), ("Overcharge / billing", 0.76, 0.55), ("No refund policy",     0.62, 0.35)],
        "performance": [("Slow load times",       0.61, 0.50), ("App crashes",          0.55, 0.60), ("Search latency",       0.42, 0.40)],
        "onboarding":  [("Complex setup",         0.45, 0.25), ("Missing docs",         0.38, 0.20), ("Confusing UI",         0.32, 0.30)],
        "features":    [("Missing integrations",  0.40, 0.65), ("Feature gaps",         0.35, 0.55), ("Poor mobile UX",       0.28, 0.45)],
        "security":    [("Data privacy concerns", 0.52, 0.90), ("Login issues",         0.30, 0.70), ("Audit log gaps",       0.22, 0.85)],
    }

    scorecard = []
    priority_matrix = []
    total_rev_risk = 0

    for attr in ATTRS:
        cfg = ATTR_CFG[attr]
        d   = attr_data[attr]

        if d["count"] > 0:
            neg_rate = round(d["neg"] / d["count"], 3)
            score    = max(0, min(100, round(100 - neg_rate * 100)))
        else:
            neg_rate = round(cfg["neg"] + rng.uniform(-0.03, 0.03), 3)
            score    = cfg["score"] + rng.randint(-2, 2)

        status = "critical" if score < 40 else "warn" if score < 60 else "ok"

        delta = rng.choice([-2, -1, -1, 0, 1]) * rng.randint(1, 4)
        trend = "improving" if delta > 0 else "worsening" if delta < 0 else "stable"

        churn_risk   = round(cfg["churn"] + rng.uniform(-0.04, 0.04), 3)
        neg_signals  = max(d["neg"], round(neg_rate * 80))
        attr_rev     = round(neg_signals * churn_risk * cfg["clv"] / 10)
        total_rev_risk += attr_rev

        scorecard.append({
            "attribute":       attr,
            "score":           score,
            "status":          status,
            "trend":           trend,
            "trend_delta":     abs(delta),
            "negative_rate":   neg_rate,
            "churn_risk":      churn_risk,
            "revenue_at_risk": attr_rev,
            "recommendation":  RECOMMENDATIONS[attr],
            "colour":          ATTR_COLOUR[attr],
        })

        for issue_name, impact, effort in ATTR_ISSUES.get(attr, []):
            quadrant = (
                "quick_win"  if impact >= 0.5 and effort <= 0.5 else
                "strategic"  if impact >= 0.5 and effort >  0.5 else
                "fill_in"    if impact <  0.5 and effort <= 0.5 else
                "avoid"
            )
            priority_matrix.append({
                "issue":     issue_name,
                "attribute": attr,
                "impact":    impact,
                "effort":    effort,
                "quadrant":  quadrant,
                "colour":    ATTR_COLOUR[attr],
            })

    scorecard.sort(key=lambda x: x["score"])   # worst first

    overall_health    = round(sum(s["score"]      for s in scorecard) / len(scorecard))
    churn_risk_index  = round(sum(s["churn_risk"] for s in scorecard) / len(scorecard), 3)
    dissatisfied_pct  = sum(s["negative_rate"]    for s in scorecard) / len(scorecard)

    total_customers   = 12_400
    dissatisfied      = round(total_customers * dissatisfied_pct)
    churn_at_risk     = round(dissatisfied    * churn_risk_index)
    avg_clv           = 3_000

    revenue_model = {
        "total_customers":    total_customers,
        "dissatisfied_count": dissatisfied,
        "churn_risk_count":   churn_at_risk,
        "revenue_at_risk":    churn_at_risk * avg_clv,
        "avg_clv":            avg_clv,
        "dissatisfied_pct":   round(dissatisfied_pct * 100, 1),
        "churn_risk_pct":     round(churn_at_risk / total_customers * 100, 1),
    }

    action_plan = [
        {"rank": 1, "action": "Real-time reliability monitoring & auto-escalation",  "attribute": "reliability", "owner": "Engineering", "timeline": "2 weeks", "impact": "+12 health pts", "priority": "CRITICAL"},
        {"rank": 2, "action": "Billing transparency redesign — eliminate hidden fees", "attribute": "pricing",     "owner": "Finance",     "timeline": "3 weeks", "impact": "+9 health pts",  "priority": "CRITICAL"},
        {"rank": 3, "action": "Support staffing + AI triage chatbot first response",   "attribute": "support",     "owner": "CX Team",     "timeline": "4 weeks", "impact": "+8 health pts",  "priority": "HIGH"},
        {"rank": 4, "action": "CDN + caching — target p95 latency < 200ms",           "attribute": "performance", "owner": "Platform",    "timeline": "3 weeks", "impact": "+5 health pts",  "priority": "HIGH"},
        {"rank": 5, "action": "Guided onboarding flow with milestone nudge emails",    "attribute": "onboarding",  "owner": "Product",     "timeline": "6 weeks", "impact": "+4 health pts",  "priority": "MEDIUM"},
    ]

    return {
        "brand_scorecard":       scorecard,
        "priority_matrix":       priority_matrix,
        "revenue_model":         revenue_model,
        "action_plan":           action_plan,
        "overall_health":        overall_health,
        "churn_risk_index":      churn_risk_index,
        "total_revenue_at_risk": total_rev_risk,
    }
