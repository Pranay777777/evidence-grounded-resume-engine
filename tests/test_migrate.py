"""Migrations build exactly what the models describe."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Engine, create_engine

from grounded.config import get_settings
from grounded.evidence.models import Base
from grounded.migrate import current, ensure_schema, head, main
from schema_shape import shape


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    return create_engine(f"sqlite:///{tmp_path / 'store.db'}")


def test_migrations_match_the_models(engine: Engine, tmp_path: Path) -> None:
    reference = create_engine(f"sqlite:///{tmp_path / 'reference.db'}")
    Base.metadata.create_all(reference)
    ensure_schema(engine)
    assert shape(engine) == shape(reference)


def test_no_model_change_is_missing_a_migration(engine: Engine) -> None:
    ensure_schema(engine)
    with engine.connect() as conn:
        context = MigrationContext.configure(conn, opts={"compare_type": True})
        assert compare_metadata(context, Base.metadata) == []


def test_upgrading_twice_is_a_no_op(engine: Engine) -> None:
    ensure_schema(engine)
    assert ensure_schema(engine) == head() == current(engine)


def test_check_reports_pending_work(
    engine: Engine, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATABASE_URL", str(engine.url))
    get_settings.cache_clear()
    try:
        assert main(["--check"]) == 1
        assert main([]) == 0
        assert main(["--check"]) == 0
    finally:
        get_settings.cache_clear()
    assert "evidence store at" in capsys.readouterr().out
