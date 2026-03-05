"""
apps/api/app/core/store.py

Lightweight in-memory store that accumulates predictions made during the
current server session.  Feeds the analytics endpoints so the Overview
dashboard reflects real demo data instead of static mock numbers.

Reset on server restart (intentional — this is a demo store, not a DB).
"""
from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import TypedDict


class PredictionRecord(TypedDict):
    ts: str                    # ISO timestamp
    sentiment: str             # positive | neutral | negative
    confidence: float
    urgency: str               # critical | high | medium | low
    severity_score: float
    business_impact_score: float
    attributes: list[str]


class _Store:
    """Thread-safe, append-only prediction store."""

    def __init__(self, maxlen: int = 50_000):
        self._lock = threading.Lock()
        self._records: list[PredictionRecord] = []
        self._maxlen = maxlen

    def append(self, record: PredictionRecord) -> None:
        with self._lock:
            self._records.append(record)
            if len(self._records) > self._maxlen:
                self._records = self._records[-self._maxlen:]

    def snapshot(self) -> list[PredictionRecord]:
        with self._lock:
            return list(self._records)

    def today(self) -> list[PredictionRecord]:
        today = datetime.now(tz=timezone.utc).date().isoformat()
        with self._lock:
            return [r for r in self._records if r["ts"].startswith(today)]

    def last_n_days(self, days: int) -> list[PredictionRecord]:
        from datetime import timedelta
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days)).isoformat()
        with self._lock:
            return [r for r in self._records if r["ts"] >= cutoff]

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    @property
    def total(self) -> int:
        with self._lock:
            return len(self._records)


# Module-level singleton — imported by predict.py and analytics.py
prediction_store = _Store()
