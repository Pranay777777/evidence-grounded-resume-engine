"""tenancy

Adds `tenant_id` to every evidence-store table (existing rows join the
`default` tenant) and, on Postgres, row-level security policies that limit
each transaction to the tenant in `app.tenant_id` (ADR-017).

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-29
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = ("role", "project", "evidence", "evidence_embedding")
POLICY = "tenant_isolation"


def upgrade() -> None:
    for table in TABLES:
        with op.batch_alter_table(table) as batch:
            batch.add_column(
                sa.Column(
                    "tenant_id", sa.String(length=40), server_default="default", nullable=False
                )
            )
        op.create_index(f"ix_{table}_tenant_id", table, ["tenant_id"])
    if op.get_bind().dialect.name == "postgresql":
        for table in TABLES:
            op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
            op.execute(
                f"CREATE POLICY {POLICY} ON {table} "
                "USING (tenant_id = current_setting('app.tenant_id', true)) "
                "WITH CHECK (tenant_id = current_setting('app.tenant_id', true))"
            )


def downgrade() -> None:
    postgres = op.get_bind().dialect.name == "postgresql"
    for table in reversed(TABLES):
        if postgres:
            op.execute(f"DROP POLICY IF EXISTS {POLICY} ON {table}")
            op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
        op.drop_index(f"ix_{table}_tenant_id", table_name=table)
        with op.batch_alter_table(table) as batch:
            batch.drop_column("tenant_id")
