"""Bring the evidence store to the current schema.

    python -m grounded.migrate            # upgrade to head
    python -m grounded.migrate --check    # exit 1 if anything is pending

The config needs no alembic.ini on disk, so it works from an installed
wheel as well as a checkout (the lakehouse's ADR-020 pattern).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from importlib.resources import files
from typing import Any

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, create_engine

from grounded.config import get_settings


def for_dialect(dialect: str) -> Callable[..., bool]:
    """An autogenerate filter that honours `ddl_if(dialect=...)`.

    Alembic compares every index in the metadata, including ones declared
    for another database only — so on SQLite it would report the Postgres
    HNSW index as missing. This skips schema objects that would not exist on
    `dialect` in the first place.
    """

    def include(obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any) -> bool:
        ddl_if = getattr(obj, "_ddl_if", None)
        return not (not reflected and ddl_if is not None and ddl_if.dialect not in (None, dialect))

    return include


def alembic_config(engine: Engine) -> Config:
    config = Config()
    config.set_main_option("script_location", str(files("grounded") / "migrations"))
    config.set_main_option(
        "sqlalchemy.url", engine.url.render_as_string(hide_password=False).replace("%", "%%")
    )
    return config


def head() -> str:
    script = ScriptDirectory.from_config(alembic_config(create_engine("sqlite://")))
    revision = script.get_current_head()
    assert revision is not None
    return revision


def current(engine: Engine) -> str | None:
    with engine.connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def ensure_schema(engine: Engine) -> str:
    """Upgrade to head; returns the revision the database ends at."""
    command.upgrade(alembic_config(engine), "head")
    at = current(engine)
    assert at is not None
    return at


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="grounded.migrate", description=__doc__)
    parser.add_argument("--check", action="store_true", help="report without applying")
    args = parser.parse_args(argv)
    engine = create_engine(get_settings().database_url)
    target = head()
    if args.check:
        at = current(engine)
        print(f"database at {at or 'empty'}, head is {target}")
        return 0 if at == target else 1
    print(f"evidence store at {ensure_schema(engine)} (head {target})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
