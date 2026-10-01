"""The public demo: prepare a database from the synthetic career, then serve (ADR-020).

    python -m grounded.demo           # load + index if empty, then serve
    python -m grounded.demo --warm    # download the models only (image build time)

Runs with DEMO=true: the UI and read-only APIs, no admin, no evidence writes,
per-visitor rate limits and a daily model budget. The evidence is the same
synthetic career the benchmarks use, so every number in the README can be
reproduced against what the demo shows.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from grounded.api.app import create_app
from grounded.config import Settings, get_settings
from grounded.evidence.models import Evidence
from grounded.evidence.store import load, read_file
from grounded.logging import configure_logging
from grounded.migrate import ensure_schema
from grounded.retrieval.embedding import get_embedder
from grounded.retrieval.index import embed_pending
from grounded.verification.nli import get_verifier

CORPUS = Path("benchmarks/retrieval/corpus.yaml")
log = logging.getLogger(__name__)


def warm(settings: Settings) -> None:
    """Download and load every model once, so the first visitor does not wait."""
    get_embedder(settings.embedder).embed(["warm-up"])
    get_verifier(settings.verifier, cache_dir=settings.model_cache_dir).check("a", "a")


def prepare(settings: Settings, corpus: Path = CORPUS) -> int:
    """Migrate, load the synthetic career if the store is empty, index it.
    Returns the number of records in the store."""
    engine = create_engine(settings.database_url)
    ensure_schema(engine)
    with Session(engine) as session:
        count = session.scalar(select(func.count()).select_from(Evidence)) or 0
        if count == 0:
            load(session, read_file(corpus))
            session.commit()
        embed_pending(session, get_embedder(settings.embedder))
        session.commit()
        return session.scalar(select(func.count()).select_from(Evidence)) or 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="grounded.demo",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--warm", action="store_true", help="download models and exit")
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    if args.warm:
        warm(settings)
        return 0
    if not settings.demo:
        parser.error("set DEMO=true - the demo serves read-only, rate-limited pages")
    records = prepare(settings)
    log.info("demo ready with %d records", records)
    uvicorn.run(create_app(), host=settings.host, port=settings.port, log_config=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
