"""The FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI
from sqlalchemy import Engine, create_engine

from grounded import __version__
from grounded.api import admin, routes
from grounded.config import get_settings


def create_app(engine: Engine | None = None) -> FastAPI:
    """Build the app. Tests pass an engine; the server builds one from settings."""
    app = FastAPI(
        title="evidence-grounded-resume-engine",
        version=__version__,
        description="Evidence store API. Every claim the generator makes must cite "
        "a verified record from here (ADR-001).",
    )
    app.state.engine = engine or create_engine(get_settings().database_url)
    app.include_router(routes.router)
    app.include_router(admin.router)
    return app
