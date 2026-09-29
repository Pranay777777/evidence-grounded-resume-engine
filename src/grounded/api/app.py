"""The FastAPI application."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi.extension import _rate_limit_exceeded_handler
from sqlalchemy import Engine, create_engine

from grounded import __version__
from grounded.api import admin, drafts, routes
from grounded.api.guards import CircuitBreaker, Ledger
from grounded.config import get_settings
from grounded.verification.nli import Verifier, get_verifier


def create_app(
    engine: Engine | None = None,
    client_factory: Callable[..., Any] | None = None,
    verifier: Callable[[], Verifier | None] | None = None,
) -> FastAPI:
    """Build the app. Tests pass an engine, a fake client and a fake verifier;
    the server builds everything from settings."""
    settings = get_settings()
    app = FastAPI(
        title="evidence-grounded-resume-engine",
        version=__version__,
        description="Evidence store API, and a generation API whose every claim cites "
        "verified evidence and passes the grounding gate (ADR-001, ADR-016). "
        "Generation endpoints need an `X-API-Key` header.",
    )
    app.state.engine = engine or create_engine(settings.database_url)
    app.state.limiter = drafts.limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    app.state.ledger = Ledger()
    app.state.breaker = CircuitBreaker(settings.circuit_failures, settings.circuit_cooldown_s)
    app.state.client_factory = client_factory or drafts.default_client_factory

    @lru_cache(maxsize=1)
    def load_verifier() -> Verifier | None:
        return get_verifier(settings.verifier, cache_dir=settings.model_cache_dir)

    app.state.get_verifier = verifier or load_verifier
    app.include_router(routes.router)
    app.include_router(drafts.router)
    app.include_router(admin.router)
    return app
