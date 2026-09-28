"""Alembic environment: the models are the target, settings give the URL."""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine, pool

from grounded.config import get_settings
from grounded.evidence.models import Base
from grounded.migrate import for_dialect

target_metadata = Base.metadata


def _url() -> str:
    # A URL set on the config wins — tests and ensure_schema use that.
    configured = context.config.get_main_option("sqlalchemy.url")
    return configured or get_settings().database_url


def _run(connection: object) -> None:
    context.configure(
        connection=connection,  # type: ignore[arg-type]
        target_metadata=target_metadata,
        render_as_batch=True,
        compare_type=True,
        include_object=for_dialect(connection.dialect.name),  # type: ignore[attr-defined]
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    engine = create_engine(_url(), poolclass=pool.NullPool)
    with engine.connect() as conn:
        _run(conn)
