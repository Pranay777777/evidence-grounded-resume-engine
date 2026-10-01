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


CORPUS = Path(__file__).parent / "fixtures" / "retrieval_corpus.yaml"


def test_pgvector_dense_search_agrees_with_plain_cosine(pg: Engine) -> None:
    from grounded.evidence.models import EvidenceEmbedding
    from grounded.retrieval.embedding import HashingEmbedder
    from grounded.retrieval.index import embed_pending
    from grounded.retrieval.search import Mode, search

    embedder = HashingEmbedder()
    query = "Delta Lake incremental loads and pytest suites"
    ensure_schema(pg)
    with Session(pg) as s:
        load(s, read_file(CORPUS))
        embed_pending(s, embedder)
        via_pgvector = [h.evidence_id for h in search(s, query, embedder, mode=Mode.DENSE, k=4)]
        vector = embedder.embed([query])[0]
        rows = s.query(EvidenceEmbedding).all()
        by_python = [
            r.evidence_id
            for r in sorted(
                rows,
                key=lambda r: (
                    -sum(a * b for a, b in zip(r.embedding, vector, strict=True)),
                    r.evidence_id,
                ),
            )
        ]
    assert via_pgvector == by_python[:4]


def test_the_hnsw_index_exists_and_is_used(pg: Engine) -> None:
    ensure_schema(pg)
    with pg.connect() as conn:
        method: str = conn.execute(
            text(
                "SELECT am.amname FROM pg_class c JOIN pg_am am ON am.oid = c.relam "
                "WHERE c.relname = 'ix_evidence_embedding_hnsw'"
            )
        ).scalar_one()
        conn.execute(text("SET enable_seqscan = off"))
        plan = "\n".join(
            conn.execute(
                text(
                    "EXPLAIN SELECT evidence_id FROM evidence_embedding "
                    "ORDER BY embedding <=> (SELECT array_fill(0.1, ARRAY[384])::vector) LIMIT 5"
                )
            ).scalars()
        )
    assert method == "hnsw"
    assert "ix_evidence_embedding_hnsw" in plan


def test_row_level_security_isolates_tenants_below_the_orm(pg: Engine) -> None:
    """Defence in depth (ADR-017): raw SQL, no ORM filter, a non-superuser role -
    Postgres itself still returns only the transaction's tenant."""
    from grounded.evidence.schema import EvidenceIn
    from grounded.evidence.store import upsert_evidence
    from grounded.evidence.tenancy import scope

    ensure_schema(pg)
    for tenant in ("acme", "globex"):
        with scope(Session(pg), tenant) as session:
            item = {
                "id": f"{tenant}-fact",
                "kind": "achievement",
                "statement": f"Built the {tenant} pipeline for the RLS test.",
            }
            upsert_evidence(session, EvidenceIn.model_validate(item))
            session.commit()

    role = f"rls_probe_{uuid.uuid4().hex[:8]}"
    with pg.connect() as conn:
        conn.execute(text(f'CREATE ROLE "{role}" NOLOGIN'))
        conn.execute(text(f'GRANT SELECT, INSERT ON evidence TO "{role}"'))
        conn.commit()
    try:
        with pg.connect() as conn:
            conn.execute(text(f'SET ROLE "{role}"'))
            conn.execute(text("SELECT set_config('app.tenant_id', 'acme', false)"))
            assert conn.execute(text("SELECT id FROM evidence")).scalars().all() == ["acme-fact"]
            conn.execute(text("SELECT set_config('app.tenant_id', 'globex', false)"))
            assert conn.execute(text("SELECT id FROM evidence")).scalars().all() == ["globex-fact"]
            conn.execute(text("SELECT set_config('app.tenant_id', '', false)"))
            assert conn.execute(text("SELECT id FROM evidence")).scalars().all() == []
            # Writing a row for another tenant is refused by the policy's WITH CHECK.
            conn.execute(text("SELECT set_config('app.tenant_id', 'acme', false)"))
            with pytest.raises(Exception, match="row-level security"):
                conn.execute(
                    text(
                        "INSERT INTO evidence (id, kind, statement, content_hash, tenant_id, "
                        "verification_status, revision, skills, created_at, updated_at) VALUES "
                        "('sneaky', 'achievement', 'Planted into another tenant.', 'h', 'globex', "
                        "'unverified', 1, '[]', now(), now())"
                    )
                )
            conn.rollback()
            conn.execute(text("RESET ROLE"))
    finally:
        with pg.connect() as conn:
            conn.execute(text(f'REVOKE ALL ON evidence FROM "{role}"'))
            conn.execute(text(f'DROP ROLE "{role}"'))
            conn.commit()


def test_scoped_sessions_set_the_tenant_for_postgres(pg: Engine) -> None:
    from grounded.evidence.tenancy import scope

    ensure_schema(pg)
    with scope(Session(pg), "acme") as session:
        value = session.execute(text("SELECT current_setting('app.tenant_id', true)")).scalar()
        assert value == "acme"
