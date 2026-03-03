"""
apps/api/app/api/v1/endpoints/reports.py

Generates intelligence reports from the live prediction store.
Reports are created on demand and returned as structured JSON.
No DB required — all data comes from the in-memory prediction store.
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.store import prediction_store

router = APIRouter()

# In-memory report registry (keyed by report_id)
_report_registry: dict[str, dict] = {}


class GenerateReportRequest(BaseModel):
    type: str  # "executive_summary" | "drift_analysis" | "cluster_intelligence" | "risk_breakdown"
    title: str | None = None


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _now_display() -> str:
    return datetime.now(tz=timezone.utc).strftime("%d %b %Y, %H:%M UTC")


# ─── Report generators ────────────────────────────────────────────────────────

def _executive_summary() -> dict[str, Any]:
    records = prediction_store.snapshot()
    total = len(records)

    if total == 0:
        return {
            "note": "No predictions recorded yet. Run the Live Stream demo to generate data.",
            "sections": [],
        }

    pos  = sum(1 for r in records if r["sentiment"] == "positive")
    neg  = sum(1 for r in records if r["sentiment"] == "negative")
    neu  = total - pos - neg
    crit = sum(1 for r in records if r["urgency"] == "critical")
    high = sum(1 for r in records if r["urgency"] == "high")
    avg_conf   = sum(r["confidence"] for r in records) / total
    avg_impact = sum(r["business_impact_score"] for r in records) / total

    health_score = round((pos / total) * 100, 1)
    neg_rate     = round(neg / total, 4)

    # Top attributes by frequency
    attr_counts: dict[str, int] = defaultdict(int)
    for r in records:
        for a in r.get("attributes", []):
            attr_counts[a] += 1
    top_attrs = sorted(attr_counts.items(), key=lambda x: -x[1])[:5]

    return {
        "sections": [
            {
                "title": "Sentiment Health",
                "kpis": [
                    {"label": "Health Score",      "value": f"{health_score}",  "unit": "/100",  "status": "good" if health_score >= 60 else "risk"},
                    {"label": "Positive Rate",     "value": f"{pos/total:.1%}", "unit": "",      "status": "good"},
                    {"label": "Negative Rate",     "value": f"{neg_rate:.1%}",  "unit": "",      "status": "risk" if neg_rate > 0.30 else "neutral"},
                    {"label": "Avg Confidence",    "value": f"{avg_conf:.1%}",  "unit": "",      "status": "neutral"},
                    {"label": "Avg Business Impact","value": f"{avg_impact:.2f}","unit": "/1.00","status": "neutral"},
                ],
                "insight": (
                    f"Sentiment health at {health_score}/100. "
                    f"{'Negative rate exceeds 30% — requires executive attention.' if neg_rate > 0.30 else 'Negative rate within acceptable band.'} "
                    f"Model confidence averaging {avg_conf:.1%} indicates reliable signal quality."
                ),
            },
            {
                "title": "Risk Exposure",
                "kpis": [
                    {"label": "Critical Signals", "value": str(crit), "unit": "messages", "status": "risk" if crit > 0 else "good"},
                    {"label": "High Priority",    "value": str(high), "unit": "messages", "status": "warn" if high > 5 else "neutral"},
                    {"label": "Escalation Rate",  "value": f"{(crit+high)/total:.1%}", "unit": "", "status": "risk" if (crit+high)/total > 0.15 else "neutral"},
                ],
                "insight": (
                    f"{crit} critical and {high} high-priority signals detected. "
                    f"Escalation rate of {(crit+high)/total:.1%} "
                    f"{'is above the 15% threshold — immediate triage required.' if (crit+high)/total > 0.15 else 'remains within operational tolerance.'}"
                ),
            },
            {
                "title": "Top Issue Areas",
                "attributes": [{"attribute": a, "count": c, "share": f"{c/total:.1%}"} for a, c in top_attrs],
                "insight": (
                    f"Primary friction areas: {', '.join(a for a, _ in top_attrs[:3])}. "
                    f"Targeted resolution in these domains offers the highest NPS leverage."
                ),
            },
            {
                "title": "Volume Summary",
                "kpis": [
                    {"label": "Total Analysed", "value": str(total), "unit": "messages", "status": "neutral"},
                    {"label": "Positive",       "value": str(pos),   "unit": "",          "status": "good"},
                    {"label": "Neutral",        "value": str(neu),   "unit": "",          "status": "neutral"},
                    {"label": "Negative",       "value": str(neg),   "unit": "",          "status": "risk"},
                ],
                "insight": f"{total} messages processed this session. Distribution: {pos/total:.0%} positive, {neu/total:.0%} neutral, {neg/total:.0%} negative.",
            },
        ],
    }


def _risk_breakdown() -> dict[str, Any]:
    records = prediction_store.snapshot()
    total = len(records)

    if total == 0:
        return {"note": "No data yet.", "tiers": []}

    tier_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    urgency_map = {"critical": "CRITICAL", "high": "HIGH", "medium": "MEDIUM", "low": "LOW"}

    for r in records:
        tier = urgency_map.get(r["urgency"], "LOW")
        tier_counts[tier] += 1

    avg_severity = sum(r["severity_score"] for r in records) / total
    avg_impact   = sum(r["business_impact_score"] for r in records) / total

    return {
        "total": total,
        "avg_severity": round(avg_severity, 3),
        "avg_business_impact": round(avg_impact, 3),
        "tiers": [
            {
                "tier": tier,
                "count": count,
                "share": f"{count/total:.1%}",
                "action": {
                    "CRITICAL": "Escalate within 1 hour to senior team.",
                    "HIGH":     "Respond within 4 hours, flag in weekly review.",
                    "MEDIUM":   "Address within 48 hours in next sprint.",
                    "LOW":      "Log for monthly trend report.",
                }[tier],
            }
            for tier, count in tier_counts.items()
        ],
        "insight": (
            f"Risk profile: {tier_counts['CRITICAL']} critical, {tier_counts['HIGH']} high. "
            f"Average severity {avg_severity:.2f}/1.00 with business impact averaging {avg_impact:.2f}. "
            f"{'Immediate attention required across critical tier.' if tier_counts['CRITICAL'] > 0 else 'No critical signals — maintain monitoring cadence.'}"
        ),
    }


def _cluster_intelligence() -> dict[str, Any]:
    records = prediction_store.snapshot()
    total = len(records)

    if total == 0:
        return {"note": "No data yet.", "clusters": []}

    # Group by top attribute
    clusters: dict[str, list] = defaultdict(list)
    unattributed = []
    for r in records:
        attrs = r.get("attributes", [])
        if attrs:
            clusters[attrs[0]].append(r)
        else:
            unattributed.append(r)

    cluster_rows = []
    for attr, recs in sorted(clusters.items(), key=lambda x: -len(x[1])):
        n = len(recs)
        neg = sum(1 for r in recs if r["sentiment"] == "negative")
        crit = sum(1 for r in recs if r["urgency"] == "critical")
        avg_imp = sum(r["business_impact_score"] for r in recs) / n
        cluster_rows.append({
            "theme": attr.title().replace("_", " "),
            "count": n,
            "share": f"{n/total:.1%}",
            "negative_rate": f"{neg/n:.1%}",
            "critical_count": crit,
            "avg_impact": round(avg_imp, 2),
            "dominant_sentiment": (
                "negative" if neg/n > 0.5 else "positive" if neg/n < 0.25 else "neutral"
            ),
        })

    top = cluster_rows[0] if cluster_rows else None
    insight = (
        f"Largest cluster: '{top['theme']}' ({top['count']} signals, {top['negative_rate']} negative rate). "
        f"Highest-impact domain for immediate resolution focus."
        if top else "Insufficient data for cluster analysis."
    )

    return {"total": total, "clusters": cluster_rows[:8], "insight": insight}


def _drift_analysis() -> dict[str, Any]:
    records = prediction_store.snapshot()
    total = len(records)

    if total < 10:
        return {
            "note": "Drift analysis requires at least 10 predictions. Keep the Live Stream running.",
            "psi_score": None,
        }

    # Compute a simple rolling drift proxy: compare first half vs second half confidence distribution
    half = total // 2
    first_half  = records[:half]
    second_half = records[half:]

    f_neg = sum(r["sentiment"] == "negative" for r in first_half) / half
    s_neg = sum(r["sentiment"] == "negative" for r in second_half) / (total - half)
    f_conf = sum(r["confidence"] for r in first_half) / half
    s_conf = sum(r["confidence"] for r in second_half) / (total - half)

    # Simple PSI proxy
    eps = 1e-6
    psi = abs((s_neg - f_neg) / max(f_neg, eps)) * 0.3
    psi = round(psi, 4)

    drift_detected = psi > 0.20

    return {
        "total_analysed": total,
        "psi_score": psi,
        "drift_detected": drift_detected,
        "alert": psi > 0.25,
        "first_half": {
            "n": half,
            "negative_rate": round(f_neg, 4),
            "avg_confidence": round(f_conf, 4),
        },
        "second_half": {
            "n": total - half,
            "negative_rate": round(s_neg, 4),
            "avg_confidence": round(s_conf, 4),
        },
        "insight": (
            f"PSI score: {psi:.4f}. "
            f"{'⚠ Drift detected — negative rate shifted {:.1%} between session halves. Review model calibration.'.format(abs(s_neg - f_neg)) if drift_detected else 'Distribution stable across session. Model performing consistently.'}"
        ),
    }


# ─── Endpoint handlers ────────────────────────────────────────────────────────

REPORT_GENERATORS = {
    "executive_summary":    ("Executive Summary",        _executive_summary),
    "risk_breakdown":       ("Risk Breakdown",            _risk_breakdown),
    "cluster_intelligence": ("Cluster Intelligence",     _cluster_intelligence),
    "drift_analysis":       ("Drift & Stability Report", _drift_analysis),
}

DEFAULT_TITLES = {k: v[0] for k, v in REPORT_GENERATORS.items()}


@router.get("", summary="List all generated reports")
async def list_reports():
    reports = [
        {
            "id": rid,
            "title": meta["title"],
            "type": meta["type"],
            "status": "ready",
            "created_at": meta["created_at"],
            "record_count": meta["record_count"],
        }
        for rid, meta in sorted(
            _report_registry.items(),
            key=lambda x: x[1]["created_at"],
            reverse=True,
        )
    ]
    return {"reports": reports, "total": len(reports)}


@router.post("", summary="Generate a new report")
async def generate_report(body: GenerateReportRequest):
    if body.type not in REPORT_GENERATORS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown report type '{body.type}'. Valid: {list(REPORT_GENERATORS)}",
        )

    title, generator_fn = REPORT_GENERATORS[body.type]
    title = body.title or title

    report_id = str(uuid.uuid4())[:8]
    data = generator_fn()
    record_count = prediction_store.total

    report = {
        "id": report_id,
        "title": title,
        "type": body.type,
        "status": "ready",
        "created_at": _now_iso(),
        "created_at_display": _now_display(),
        "record_count": record_count,
        "data": data,
    }

    _report_registry[report_id] = report
    return report


@router.get("/{report_id}", summary="Get a specific report by ID")
async def get_report(report_id: str):
    if report_id not in _report_registry:
        raise HTTPException(status_code=404, detail="Report not found")
    return _report_registry[report_id]


@router.delete("/{report_id}", summary="Delete a report")
async def delete_report(report_id: str):
    if report_id not in _report_registry:
        raise HTTPException(status_code=404, detail="Report not found")
    del _report_registry[report_id]
    return {"deleted": report_id}
