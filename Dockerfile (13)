"""
apps/api/app/main.py

Sentient Intelligence Platform — FastAPI entry point.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import RedactingFilter, get_settings
from app.api.v1.router import api_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logging.getLogger().addFilter(RedactingFilter())
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentry (optional)
# ---------------------------------------------------------------------------
if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.APP_ENV,
        traces_sample_rate=0.2,
    )

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.API_RATE_LIMIT_PER_MINUTE}/minute"])

# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate config at boot — fail fast before accepting traffic
    try:
        settings.validate_required()
    except RuntimeError as exc:
        logger.critical("Startup configuration error:\n%s", exc)
        raise

    logger.info(
        "Sentient API starting — env=%s  version=%s",
        settings.APP_ENV,
        settings.APP_VERSION,
    )
    yield
    logger.info("Sentient API shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title="Sentient Intelligence Platform",
        description=(
            "Enterprise-grade customer sentiment and branding intelligence. "
            "Multi-tenant · RBAC · SageMaker-backed NLP · Real-time analytics."
        ),
        version=settings.APP_VERSION,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── State ────────────────────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Middleware ───────────────────────────────────────────────────────────
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing ───────────────────────────────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        t0 = time.perf_counter()
        response: Response = await call_next(request)
        elapsed = round((time.perf_counter() - t0) * 1000)
        response.headers["X-Process-Time-Ms"] = str(elapsed)
        response.headers["X-App-Version"] = settings.APP_VERSION
        return response

    # ── Tenant context injection ─────────────────────────────────────────────
    @app.middleware("http")
    async def inject_tenant_header(request: Request, call_next):
        # Downstream handlers read request.state.tenant_id
        request.state.tenant_id = request.headers.get("X-Tenant-Id", "")
        return await call_next(request)

    # ── Routes ───────────────────────────────────────────────────────────────
    app.include_router(api_router, prefix="/api/v1")

    # ── Health ───────────────────────────────────────────────────────────────
    @app.get("/health", tags=["ops"], include_in_schema=False)
    async def health():
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "environment": settings.APP_ENV,
        }

    @app.get("/", include_in_schema=False)
    async def root():
        return {"service": "Sentient Intelligence Platform", "version": settings.APP_VERSION}

    return app


app = create_app()
