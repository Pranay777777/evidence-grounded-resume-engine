"""The evidence store on Postgres with pgvector — the engine it actually runs on.

    GROUNDED_TEST_POSTGRES_URL=postgresql+psycopg://app:app@localhost:5433/app

Each test gets a throwaway database, so pointing this at the compose stack
never touches the `app` database. Skipped when the variable is unset.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from grounded.evidence.models import Base, Evidence
from grounded.evidence.store import citable, load, read_file
from grounded.migrate import ensure_schema, head
from schema_shape import shape

ADMIN_URL = os.environ.get("GROUNDED_TEST_POSTGRES_URL", "")
FIXTURE = Path(__file__).parent / "fixtures" / "example_evidence.yaml"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ADMIN_URL, reason="GROUNDED_TEST_POSTGRES_URL is not set"),
]


def _scratch() -> Iterator[Engine]:
    admin = create_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    name = f"grounded_test_{uuid.uuid4().hex[:10]}"
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    engine = create_engine(make_url(ADMIN_URL).set(database=name))
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture
def pg() -> Iterator[Engine]:
    yield from _scratch()


@pytest.fixture
def reference() -> Iterator[Engine]:
    yield from _scratch()


def test_migrations_enable_pgvector(pg: Engine) -> None:
    assert ensure_schema(pg) == head()
    with pg.connect() as conn:
        assert (
            conn.execute(
                text("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
            ).scalar_one()
            == 1
        )
        # Prove it works, not merely that it is installed.
        distance: float = conn.execute(
            text("SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector")
        ).scalar_one()
    assert distance == pytest.approx(1.0)


def test_migrations_match_the_models_on_postgres(pg: Engine, reference: Engine) -> None:
    Base.metadata.create_all(reference)
    ensure_schema(pg)
    assert shape(pg) == shape(reference)


def test_loading_on_postgres(pg: Engine) -> None:
    ensure_schema(pg)
    with Session(pg) as s:
        report = load(s, read_file(FIXTURE))
        assert len(report.created) == 3
        assert [e.id for e in citable(s)] == ["example-pipeline-latency"]


def test_postgres_enforces_the_metric_rule(pg: Engine) -> None:
    ensure_schema(pg)
    with Session(pg) as s:
        s.add(
            Evidence(
                id="raw",
                kind="metric",
                statement="A number, supposedly.",
                content_hash="x",
                revision=1,
                verification_status="unverified",
                skills=[],
            )
        )
        with pytest.raises(IntegrityError):
            s.commit()
