"""Typed application settings, loaded from the environment."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every configurable value lives here — never read os.environ directly."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["local", "ci", "staging", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    database_url: str = "postgresql+psycopg://app:app@localhost:5433/app"
    """The local compose database. Port 5433, not 5432, so this stack and the
    lakehouse's can run side by side; the driver is named so SQLAlchemy never
    has to guess between psycopg2 and psycopg 3."""


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
