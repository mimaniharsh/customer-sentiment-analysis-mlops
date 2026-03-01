"""
apps/api/app/models/db.py

SQLAlchemy async models for the Sentient multi-tenant SaaS platform.
Schema: tenants → users/roles → messages → predictions → metrics
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class Base(AsyncAttrs, DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class SentimentLabel(str, enum.Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class UrgencyLevel(str, enum.Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class UserRole(str, enum.Enum):
    OWNER = "owner"
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


class MessageSource(str, enum.Enum):
    TWEET = "tweet"
    REVIEW = "review"
    TICKET = "ticket"
    CHAT = "chat"
    FEEDBACK = "feedback"
    EMAIL = "email"
    CSV_UPLOAD = "csv_upload"
    API = "api"


# ---------------------------------------------------------------------------
# Tenant
# ---------------------------------------------------------------------------
class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    plan: Mapped[str] = mapped_column(String(32), default="starter")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    settings: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    users: Mapped[List["User"]] = relationship("User", back_populates="tenant", lazy="selectin")
    messages: Mapped[List["Message"]] = relationship("Message", back_populates="tenant")


# ---------------------------------------------------------------------------
# User & Role
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("tenant_id", "email", name="uq_tenant_user_email"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(256), nullable=False)
    full_name: Mapped[str] = mapped_column(String(256), default="")
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.VIEWER)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="users")


# ---------------------------------------------------------------------------
# Message (inbound text for analysis)
# ---------------------------------------------------------------------------
class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_tenant_created", "tenant_id", "created_at"),
        Index("ix_messages_tenant_source", "tenant_id", "source"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    external_id: Mapped[Optional[str]] = mapped_column(String(256))
    source: Mapped[MessageSource] = mapped_column(Enum(MessageSource), default=MessageSource.API)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(8), default="en")
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    prediction: Mapped[Optional["Prediction"]] = relationship("Prediction", back_populates="message", uselist=False)
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="messages")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_tenant_sentiment", "tenant_id", "sentiment"),
        Index("ix_predictions_tenant_urgency", "tenant_id", "urgency"),
        Index("ix_predictions_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    message_id: Mapped[str] = mapped_column(String(36), ForeignKey("messages.id", ondelete="CASCADE"), unique=True)

    # Core sentiment
    sentiment: Mapped[SentimentLabel] = mapped_column(Enum(SentimentLabel), nullable=False)
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=False)  # -1.0 to 1.0
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    prob_positive: Mapped[float] = mapped_column(Float, default=0.0)
    prob_neutral: Mapped[float] = mapped_column(Float, default=0.0)
    prob_negative: Mapped[float] = mapped_column(Float, default=0.0)

    # Enhanced ML outputs
    urgency: Mapped[UrgencyLevel] = mapped_column(Enum(UrgencyLevel), default=UrgencyLevel.LOW)
    urgency_score: Mapped[float] = mapped_column(Float, default=0.0)
    severity_score: Mapped[float] = mapped_column(Float, default=0.0)
    business_impact_score: Mapped[float] = mapped_column(Float, default=0.0)

    # Multi-label attributes (e.g. ["pricing", "support", "reliability"])
    attributes: Mapped[Optional[list]] = mapped_column(ARRAY(String), default=list)
    attribute_scores: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)

    # Explainability tokens
    explanation: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)

    # Cluster membership (for root-cause clustering)
    cluster_id: Mapped[Optional[int]] = mapped_column(Integer)
    embedding_id: Mapped[Optional[str]] = mapped_column(String(36))

    # Model provenance
    model_version: Mapped[str] = mapped_column(String(64), default="v1")
    endpoint_name: Mapped[str] = mapped_column(String(256), default="")
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    message: Mapped["Message"] = relationship("Message", back_populates="prediction")


# ---------------------------------------------------------------------------
# DriftRecord (model monitoring)
# ---------------------------------------------------------------------------
class DriftRecord(Base):
    __tablename__ = "drift_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    psi_score: Mapped[float] = mapped_column(Float)           # Population Stability Index
    ks_statistic: Mapped[float] = mapped_column(Float)        # Kolmogorov–Smirnov
    drift_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    reference_distribution: Mapped[dict] = mapped_column(JSONB)
    current_distribution: Mapped[dict] = mapped_column(JSONB)
    alert_triggered: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------------------
# MetricSnapshot (precomputed KPIs for dashboards)
# ---------------------------------------------------------------------------
class MetricSnapshot(Base):
    __tablename__ = "metric_snapshots"
    __table_args__ = (Index("ix_metrics_tenant_period", "tenant_id", "period_start"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    granularity: Mapped[str] = mapped_column(String(16), default="day")  # hour|day|week|month

    total_messages: Mapped[int] = mapped_column(Integer, default=0)
    positive_count: Mapped[int] = mapped_column(Integer, default=0)
    neutral_count: Mapped[int] = mapped_column(Integer, default=0)
    negative_count: Mapped[int] = mapped_column(Integer, default=0)

    avg_sentiment_score: Mapped[float] = mapped_column(Float, default=0.0)
    avg_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    avg_business_impact: Mapped[float] = mapped_column(Float, default=0.0)
    critical_urgency_count: Mapped[int] = mapped_column(Integer, default=0)

    top_attributes: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)
    source_breakdown: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)
    drift_detected: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    created_by: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"))
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    report_type: Mapped[str] = mapped_column(String(64), default="executive_summary")
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    s3_key: Mapped[Optional[str]] = mapped_column(String(1024))
    status: Mapped[str] = mapped_column(String(32), default="pending")
    payload: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
