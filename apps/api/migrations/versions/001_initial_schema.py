"""Initial schema

Revision ID: 001_initial
Revises: 
Create Date: 2025-01-01 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── tenants ──────────────────────────────────────────────────────────────
    op.create_table(
        "tenants",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("slug", sa.String(64), nullable=False, unique=True),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("plan", sa.String(32), server_default="starter"),
        sa.Column("is_active", sa.Boolean(), server_default="true"),
        sa.Column("settings", postgresql.JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_tenants_slug", "tenants", ["slug"])

    # ── users ─────────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("hashed_password", sa.String(256), nullable=False),
        sa.Column("full_name", sa.String(256), server_default=""),
        sa.Column("role", sa.String(32), server_default="viewer"),
        sa.Column("is_active", sa.Boolean(), server_default="true"),
        sa.Column("last_login", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "email", name="uq_tenant_user_email"),
    )
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"])

    # ── messages ─────────────────────────────────────────────────────────────
    op.create_table(
        "messages",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("external_id", sa.String(256)),
        sa.Column("source", sa.String(32), server_default="api"),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("language", sa.String(8), server_default="en"),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_messages_tenant_created", "messages", ["tenant_id", "created_at"])
    op.create_index("ix_messages_tenant_source", "messages", ["tenant_id", "source"])

    # ── predictions ───────────────────────────────────────────────────────────
    op.create_table(
        "predictions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("message_id", sa.String(36), sa.ForeignKey("messages.id", ondelete="CASCADE"), unique=True),
        sa.Column("sentiment", sa.String(16), nullable=False),
        sa.Column("sentiment_score", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("prob_positive", sa.Float(), server_default="0"),
        sa.Column("prob_neutral", sa.Float(), server_default="0"),
        sa.Column("prob_negative", sa.Float(), server_default="0"),
        sa.Column("urgency", sa.String(16), server_default="low"),
        sa.Column("urgency_score", sa.Float(), server_default="0"),
        sa.Column("severity_score", sa.Float(), server_default="0"),
        sa.Column("business_impact_score", sa.Float(), server_default="0"),
        sa.Column("attributes", postgresql.ARRAY(sa.String()), server_default="{}"),
        sa.Column("attribute_scores", postgresql.JSONB(), server_default="{}"),
        sa.Column("explanation", postgresql.JSONB(), server_default="{}"),
        sa.Column("cluster_id", sa.Integer()),
        sa.Column("embedding_id", sa.String(36)),
        sa.Column("model_version", sa.String(64), server_default="v1"),
        sa.Column("endpoint_name", sa.String(256), server_default=""),
        sa.Column("latency_ms", sa.Integer()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_predictions_tenant_sentiment", "predictions", ["tenant_id", "sentiment"])
    op.create_index("ix_predictions_tenant_urgency", "predictions", ["tenant_id", "urgency"])
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])

    # ── drift_records ─────────────────────────────────────────────────────────
    op.create_table(
        "drift_records",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("psi_score", sa.Float()),
        sa.Column("ks_statistic", sa.Float()),
        sa.Column("drift_detected", sa.Boolean(), server_default="false"),
        sa.Column("reference_distribution", postgresql.JSONB()),
        sa.Column("current_distribution", postgresql.JSONB()),
        sa.Column("alert_triggered", sa.Boolean(), server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── metric_snapshots ──────────────────────────────────────────────────────
    op.create_table(
        "metric_snapshots",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("granularity", sa.String(16), server_default="day"),
        sa.Column("total_messages", sa.Integer(), server_default="0"),
        sa.Column("positive_count", sa.Integer(), server_default="0"),
        sa.Column("neutral_count", sa.Integer(), server_default="0"),
        sa.Column("negative_count", sa.Integer(), server_default="0"),
        sa.Column("avg_sentiment_score", sa.Float(), server_default="0"),
        sa.Column("avg_confidence", sa.Float(), server_default="0"),
        sa.Column("avg_business_impact", sa.Float(), server_default="0"),
        sa.Column("critical_urgency_count", sa.Integer(), server_default="0"),
        sa.Column("top_attributes", postgresql.JSONB(), server_default="{}"),
        sa.Column("source_breakdown", postgresql.JSONB(), server_default="{}"),
        sa.Column("drift_detected", sa.Boolean(), server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_metrics_tenant_period", "metric_snapshots", ["tenant_id", "period_start"])

    # ── reports ───────────────────────────────────────────────────────────────
    op.create_table(
        "reports",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_by", sa.String(36), sa.ForeignKey("users.id")),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("report_type", sa.String(64), server_default="executive_summary"),
        sa.Column("period_start", sa.DateTime(timezone=True)),
        sa.Column("period_end", sa.DateTime(timezone=True)),
        sa.Column("s3_key", sa.String(1024)),
        sa.Column("status", sa.String(32), server_default="pending"),
        sa.Column("payload", postgresql.JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    for table in ["reports", "metric_snapshots", "drift_records", "predictions", "messages", "users", "tenants"]:
        op.drop_table(table)
