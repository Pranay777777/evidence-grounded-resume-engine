"""Structural fingerprint of a database schema, for comparing two of them."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Engine, inspect


def shape(engine: Engine) -> dict[str, dict[str, Any]]:
    ins = inspect(engine)
    out: dict[str, dict[str, Any]] = {}
    for table in sorted(ins.get_table_names()):
        if table == "alembic_version":
            continue
        out[table] = {
            "columns": sorted(
                (c["name"], str(c["type"]).split("(")[0].upper(), c["nullable"])
                for c in ins.get_columns(table)
            ),
            "checks": sorted(c["name"] or "?" for c in ins.get_check_constraints(table)),
            "indexes": sorted(str(i["name"]) for i in ins.get_indexes(table)),
            "foreign_keys": sorted(
                (tuple(f["constrained_columns"]), f["referred_table"])
                for f in ins.get_foreign_keys(table)
            ),
            "primary_key": sorted(ins.get_pk_constraint(table)["constrained_columns"]),
        }
    return out
