"""evidence embeddings

One vector per (record, embedder), stamped with the record revision it
encodes (ADR-004). On Postgres the column is pgvector's `vector(384)` with an
HNSW cosine index; SQLite, used by the unit tests, stores the list as JSON.

Written by hand: autogenerate cannot express the HNSW index or the
per-dialect column type.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DIM = 384


def upgrade() -> None:
    postgres = op.get_bind().dialect.name == "postgresql"
    if postgres:
        from pgvector.sqlalchemy import Vector

        vector_type: sa.types.TypeEngine[object] = Vector(DIM)
    else:
        vector_type = sa.JSON()

    op.create_table(
        "evidence_embedding",
        sa.Column("evidence_id", sa.String(length=80), nullable=False),
        sa.Column("embedder", sa.String(length=40), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("embedding", vector_type, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["evidence_id"], ["evidence.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("evidence_id", "embedder"),
    )
    if postgres:
        op.create_index(
            "ix_evidence_embedding_hnsw",
            "evidence_embedding",
            ["embedding"],
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        )


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.drop_index("ix_evidence_embedding_hnsw", table_name="evidence_embedding")
    op.drop_table("evidence_embedding")
