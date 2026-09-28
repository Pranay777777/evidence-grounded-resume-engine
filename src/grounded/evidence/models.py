"""Evidence store tables.

Three tables, deliberately few:

- `role` — a position held: title, organisation, months.
- `project` — a body of work, optionally within a role.
- `evidence` — one checkable fact, optionally tied to a project or role.

Generated bullets cite `evidence.id`. Roles and projects give that
evidence context for retrieval and phrasing; they are never cited alone,
because "worked at X" is a fact only once an evidence record says so.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from grounded.evidence.enums import EvidenceKind, VerificationMethod, VerificationStatus

ID_PATTERN = r"^[a-z0-9]+(-[a-z0-9]+)*$"
"""Stable IDs are lowercase slugs: readable in a citation, safe in a URL."""

MONTH_PATTERN = r"^\d{4}-(0[1-9]|1[0-2])$"
"""Dates are months. Résumés are written at month precision, and a day
would imply knowledge the evidence rarely has."""


def _in(column: str, values: type[Any]) -> CheckConstraint:
    allowed = ", ".join(f"'{v.value}'" for v in values)
    return CheckConstraint(f"{column} IN ({allowed})", name=f"ck_{column}")


class Base(DeclarativeBase):
    pass


class Timestamps:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Role(Timestamps, Base):
    __tablename__ = "role"
    __table_args__ = (
        CheckConstraint(
            "start_month IS NULL OR end_month IS NULL OR start_month <= end_month",
            name="ck_role_months_ordered",
        ),
    )

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    organisation: Mapped[str] = mapped_column(String(200), nullable=False)
    start_month: Mapped[str | None] = mapped_column(String(7))
    end_month: Mapped[str | None] = mapped_column(String(7))
    """Null means current."""

    projects: Mapped[list[Project]] = relationship(back_populates="role")


class Project(Timestamps, Base):
    __tablename__ = "project"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    role_id: Mapped[str | None] = mapped_column(ForeignKey("role.id", ondelete="SET NULL"))
    summary: Mapped[str | None] = mapped_column(Text)
    repo_url: Mapped[str | None] = mapped_column(String(500))

    role: Mapped[Role | None] = relationship(back_populates="projects")


class Evidence(Timestamps, Base):
    __tablename__ = "evidence"
    __table_args__ = (
        _in("kind", EvidenceKind),
        _in("verification_status", VerificationStatus),
        CheckConstraint(
            "verification_method IS NULL OR verification_method IN "
            "('artifact', 'third_party', 'self_attested')",
            name="ck_verification_method",
        ),
        # Only a metric may carry a number, and a metric must carry one:
        # the generator quotes numbers from nowhere else.
        CheckConstraint(
            "(kind = 'metric') = (metric_value IS NOT NULL)", name="ck_metric_has_value"
        ),
        CheckConstraint(
            "verification_status <> 'verified' OR verification_method IS NOT NULL",
            name="ck_verified_has_method",
        ),
        CheckConstraint(
            "verification_method IS NULL OR verification_method <> 'artifact' "
            "OR artifact_url IS NOT NULL",
            name="ck_artifact_method_has_url",
        ),
        CheckConstraint("revision >= 1", name="ck_revision_positive"),
        Index("ix_evidence_status", "verification_status"),
        Index("ix_evidence_project", "project_id"),
    )

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    """Stable forever. Bullets cite it; renaming it would orphan citations."""

    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    statement: Mapped[str] = mapped_column(String(400), nullable=False)
    """One checkable fact, in plain words. If it needs 'and', it is two records."""

    project_id: Mapped[str | None] = mapped_column(ForeignKey("project.id", ondelete="SET NULL"))
    role_id: Mapped[str | None] = mapped_column(ForeignKey("role.id", ondelete="SET NULL"))
    month: Mapped[str | None] = mapped_column(String(7))

    metric_value: Mapped[Decimal | None] = mapped_column(Numeric(18, 4))
    metric_unit: Mapped[str | None] = mapped_column(String(40))
    skills: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    artifact_url: Mapped[str | None] = mapped_column(String(500))

    verification_status: Mapped[str] = mapped_column(
        String(20), default=VerificationStatus.UNVERIFIED, nullable=False
    )
    verification_method: Mapped[str | None] = mapped_column(String(20))
    verified_by: Mapped[str | None] = mapped_column(String(200))
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    revision: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    """Increments whenever the fact changes. A generation records the
    revision it cited, so an edit can never silently change what a past
    résumé claimed."""

    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    """Hash of the fields that make up the fact — how a change is detected."""

    project: Mapped[Project | None] = relationship()
    role: Mapped[Role | None] = relationship()

    @property
    def citable(self) -> bool:
        return self.verification_status == VerificationStatus.VERIFIED


__all__ = [
    "ID_PATTERN",
    "MONTH_PATTERN",
    "Base",
    "Evidence",
    "EvidenceKind",
    "Project",
    "Role",
    "VerificationMethod",
    "VerificationStatus",
]
