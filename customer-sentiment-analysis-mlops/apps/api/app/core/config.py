"""
apps/api/app/core/config.py

Runtime configuration with environment-aware secret loading.
- Local / development: reads from .env / environment variables
- Production (APP_ENV=production): fetches from AWS Secrets Manager

No secrets are embedded in this file or anywhere in the repository.
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Redacting filter — strips secrets from log output
# ---------------------------------------------------------------------------
_REDACT_PATTERNS = [
    re.compile(r"(AKIA[0-9A-Z]{16})"),           # AWS Access Key ID
    re.compile(r"([A-Za-z0-9/+=]{40})"),           # AWS Secret Key (heuristic)
    re.compile(r"(password\s*=\s*)\S+", re.I),
    re.compile(r"(secret\s*=\s*)\S+", re.I),
    re.compile(r"(token\s*=\s*)\S+", re.I),
]


class RedactingFilter(logging.Filter):
    """Removes credential patterns from log records before emission."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        msg = record.getMessage()
        for pattern in _REDACT_PATTERNS:
            msg = pattern.sub(r"[REDACTED]", msg)
        record.msg = msg
        record.args = ()
        return True


def _load_from_secrets_manager(secret_name: str, region: str) -> dict[str, Any]:
    """Fetch a JSON secret from AWS Secrets Manager."""
    try:
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response["SecretString"])
    except ClientError as exc:
        logger.error("Failed to load secret '%s': %s", secret_name, exc.response["Error"]["Code"])
        raise
    except NoCredentialsError:
        logger.error(
            "No AWS credentials found. Ensure IAM role is attached (production) "
            "or AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are set (development)."
        )
        raise


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Application ─────────────────────────────────────────────────────────
    APP_ENV: str = "development"
    APP_NAME: str = "Sentient Intelligence Platform"
    APP_VERSION: str = "2.0.0"
    APP_SECRET_KEY: str = ""
    APP_ALLOWED_ORIGINS: str = "http://localhost:3000"
    API_RATE_LIMIT_PER_MINUTE: int = 60
    LOG_LEVEL: str = "INFO"

    # ── JWT ─────────────────────────────────────────────────────────────────
    JWT_SECRET_KEY: str = ""
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── AWS ─────────────────────────────────────────────────────────────────
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "ap-south-1"
    AWS_SECRETS_MANAGER_SECRET_NAME: str = "sentient/production/secrets"

    # ── SageMaker ───────────────────────────────────────────────────────────
    SAGEMAKER_ENDPOINT_NAME: str = "customer-sentiment-analysis-mlops-endpoint"
    SAGEMAKER_ROLE_ARN: str = ""

    # ── S3 ──────────────────────────────────────────────────────────────────
    S3_BUCKET: str = ""
    S3_PROJECT_PREFIX: str = "customer-sentiment-analysis-mlops"

    # ── Database ────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://sentient:sentient_dev@localhost:5432/sentient_db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    # ── Redis ───────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CELERY_URL: str = "redis://localhost:6379/1"

    # ── Observability ───────────────────────────────────────────────────────
    SENTRY_DSN: Optional[str] = None

    # ── Multi-Tenancy ───────────────────────────────────────────────────────
    DEFAULT_TENANT_SLUG: str = "default"
    MAX_TENANTS: int = 100

    # ── Computed helpers ────────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.APP_ALLOWED_ORIGINS.split(",")]

    @field_validator("APP_ENV")
    @classmethod
    def validate_env(cls, v: str) -> str:
        allowed = {"development", "staging", "production", "test"}
        if v not in allowed:
            raise ValueError(f"APP_ENV must be one of {allowed}")
        return v

    @model_validator(mode="after")
    def overlay_secrets_manager(self) -> "Settings":
        """In production, overlay secrets from AWS Secrets Manager."""
        if self.APP_ENV == "production":
            try:
                secrets = _load_from_secrets_manager(
                    self.AWS_SECRETS_MANAGER_SECRET_NAME, self.AWS_REGION
                )
                for key, value in secrets.items():
                    if hasattr(self, key):
                        object.__setattr__(self, key, value)
                logger.info("Secrets loaded from AWS Secrets Manager.")
            except Exception as exc:
                logger.critical("Cannot start in production without Secrets Manager: %s", exc)
                raise
        return self

    def validate_required(self) -> None:
        """Raise at startup if critical variables are missing."""
        required: list[tuple[str, str]] = [
            ("APP_SECRET_KEY", "Generate with: openssl rand -hex 32"),
            ("JWT_SECRET_KEY", "Generate with: openssl rand -hex 32"),
            ("DATABASE_URL", "Set your PostgreSQL connection string"),
        ]
        missing = [
            f"  {name} — {hint}"
            for name, hint in required
            if not getattr(self, name, None)
        ]
        if missing:
            raise RuntimeError(
                "Missing required configuration:\n"
                + "\n".join(missing)
                + "\n\nSee .env.example for setup instructions."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
