# ruff: noqa: E501 — generated DDL; SQL strings do not wrap
"""evidence store baseline

Roles, projects and evidence records (ADR-002). On Postgres it also enables
pgvector, which dense retrieval needs from step 44 — enabling it here proves
the image supports it before anything depends on it. SQLite, used by the
unit tests, has no extensions and skips that line.

Revision ID: 0001
Revises:
Create Date: 2026-09-28 06:57:25.504419
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "role",
        sa.Column("id", sa.String(length=80), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("organisation", sa.String(length=200), nullable=False),
        sa.Column("start_month", sa.String(length=7), nullable=True),
        sa.Column("end_month", sa.String(length=7), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "start_month IS NULL OR end_month IS NULL OR start_month <= end_month",
            name="ck_role_months_ordered",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "project",
        sa.Column("id", sa.String(length=80), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("role_id", sa.String(length=80), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("repo_url", sa.String(length=500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["role_id"], ["role.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "evidence",
        sa.Column("id", sa.String(length=80), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("statement", sa.String(length=400), nullable=False),
        sa.Column("project_id", sa.String(length=80), nullable=True),
        sa.Column("role_id", sa.String(length=80), nullable=True),
        sa.Column("month", sa.String(length=7), nullable=True),
        sa.Column("metric_value", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("metric_unit", sa.String(length=40), nullable=True),
        sa.Column("skills", sa.JSON(), nullable=False),
        sa.Column("artifact_url", sa.String(length=500), nullable=True),
        sa.Column("verification_status", sa.String(length=20), nullable=False),
        sa.Column("verification_method", sa.String(length=20), nullable=True),
        sa.Column("verified_by", sa.String(length=200), nullable=True),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(kind = 'metric') = (metric_value IS NOT NULL)", name="ck_metric_has_value"
        ),
        sa.CheckConstraint(
            "kind IN ('achievement', 'metric', 'responsibility', 'skill', 'education', 'certification')",
            name="ck_kind",
        ),
        sa.CheckConstraint(
            "verification_method IS NULL OR verification_method <> 'artifact' OR artifact_url IS NOT NULL",
            name="ck_artifact_method_has_url",
        ),
        sa.CheckConstraint(
            "verification_method IS NULL OR verification_method IN ('artifact', 'third_party', 'self_attested')",
            name="ck_verification_method",
        ),
        sa.CheckConstraint(
            "verification_status <> 'verified' OR verification_method IS NOT NULL",
            name="ck_verified_has_method",
        ),
        sa.CheckConstraint(
            "verification_status IN ('unverified', 'verified', 'rejected')",
            name="ck_verification_status",
        ),
        sa.CheckConstraint("revision >= 1", name="ck_revision_positive"),
        sa.ForeignKeyConstraint(["project_id"], ["project.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["role_id"], ["role.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("evidence", schema=None) as batch_op:
        batch_op.create_index("ix_evidence_project", ["project_id"], unique=False)
        batch_op.create_index("ix_evidence_status", ["verification_status"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("evidence", schema=None) as batch_op:
        batch_op.drop_index("ix_evidence_status")
        batch_op.drop_index("ix_evidence_project")

    op.drop_table("evidence")
    op.drop_table("project")
    op.drop_table("role")
