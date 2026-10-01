"""Loading and verifying evidence.

The file is the source a person edits; the database is what the generator
reads. Loading is an idempotent upsert keyed on stable IDs, with one rule
that matters more than the rest:

**A changed fact is an unverified fact.** If a record's content hash
differs from what is stored, its revision increments and its verification
is cleared — whatever the file says. Verification attests to a specific
statement; editing the statement voids it. Otherwise a record verified as
"contributed to the migration" could be edited to "led the migration" and
stay citable.

Nothing is ever deleted by a load. A record missing from the file stays in
the store; removal is a deliberate act, not a side effect of a typo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from grounded.evidence.enums import VerificationMethod, VerificationStatus
from grounded.evidence.models import Evidence, Project, Role
from grounded.evidence.schema import EvidenceFile, EvidenceIn, ProjectIn, RoleIn


class EvidenceError(ValueError):
    """The file refers to something that does not exist, or breaks a rule."""


@dataclass
class LoadReport:
    created: list[str] = field(default_factory=list)
    revised: list[str] = field(default_factory=list)
    """Content changed: revision bumped, verification cleared."""
    unchanged: list[str] = field(default_factory=list)
    verified: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{len(self.created)} created, {len(self.revised)} revised "
            f"(re-verification needed), {len(self.unchanged)} unchanged, "
            f"{len(self.verified)} verified by this load"
        )


def read_file(path: Path) -> EvidenceFile:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return EvidenceFile.model_validate(raw)


def _apply_verification(record: Evidence, item: EvidenceIn, now: datetime) -> bool:
    if item.verification is None:
        return False
    record.verification_status = VerificationStatus.VERIFIED
    record.verification_method = item.verification.method
    record.verified_by = item.verification.by
    record.verified_at = now
    return True


def _write_fact(record: Evidence, item: EvidenceIn) -> None:
    record.kind = item.kind
    record.statement = item.statement.strip()
    record.project_id = item.project
    record.role_id = item.role
    record.month = item.month
    record.metric_value = item.metric.value if item.metric else None
    record.metric_unit = item.metric.unit if item.metric else None
    record.skills = list(item.skills)
    record.artifact_url = str(item.artifact_url) if item.artifact_url else None
    record.content_hash = item.content_hash()


class Outcome(StrEnum):
    CREATED = "created"
    REVISED = "revised"
    UNCHANGED = "unchanged"


def _claim(session: Session, model: type[Any], ident: str) -> None:
    """IDs are unique across tenants (ADR-017). A tenant-scoped lookup that
    finds nothing may still collide with another tenant's row; say so, rather
    than failing later on the primary key."""
    taken = session.execute(
        select(model.id).where(model.id == ident).execution_options(all_tenants=True)
    ).first()
    if taken is not None:
        raise EvidenceError(f"id '{ident}' is already in use - choose another")


def upsert_role(session: Session, r: RoleIn) -> Role:
    role = session.get(Role, r.id)
    if role is None:
        _claim(session, Role, r.id)
        role = Role(id=r.id)
    role.title, role.organisation = r.title, r.organisation
    role.start_month, role.end_month = r.start, r.end
    session.add(role)
    return role


def upsert_project(session: Session, p: ProjectIn) -> Project:
    if p.role and session.get(Role, p.role) is None:
        raise EvidenceError(f"project '{p.id}' refers to unknown role '{p.role}'")
    project = session.get(Project, p.id)
    if project is None:
        _claim(session, Project, p.id)
        project = Project(id=p.id)
    project.name, project.role_id, project.summary = p.name, p.role, p.summary
    project.repo_url = str(p.repo_url) if p.repo_url else None
    session.add(project)
    return project


def upsert_evidence(
    session: Session, item: EvidenceIn, now: datetime | None = None
) -> tuple[Evidence, Outcome, bool]:
    """The single write path for a fact — the loader and the API both use it.

    Returns the record, what happened to it, and whether this write
    verified it. Does not commit: the caller decides the transaction.
    """
    moment = now or datetime.now(UTC)
    if item.project and session.get(Project, item.project) is None:
        raise EvidenceError(f"evidence '{item.id}' refers to unknown project '{item.project}'")
    if item.role and session.get(Role, item.role) is None:
        raise EvidenceError(f"evidence '{item.id}' refers to unknown role '{item.role}'")

    record = session.get(Evidence, item.id)
    if record is None:
        _claim(session, Evidence, item.id)
        record = Evidence(id=item.id, revision=1)
        _write_fact(record, item)
        record.verification_status = VerificationStatus.UNVERIFIED
        session.add(record)
        return record, Outcome.CREATED, _apply_verification(record, item, moment)

    if record.content_hash != item.content_hash():
        _write_fact(record, item)
        record.revision += 1
        # The fact changed: whatever verified the old one does not verify
        # this one. A verification block in the same write may re-verify
        # it, and that is recorded as a fresh verification.
        record.verification_status = VerificationStatus.UNVERIFIED
        record.verification_method = record.verified_by = None
        record.verified_at = None
        return record, Outcome.REVISED, _apply_verification(record, item, moment)

    record.skills = list(item.skills)  # tags are not part of the fact
    verified = not record.citable and _apply_verification(record, item, moment)
    return record, Outcome.UNCHANGED, verified


def load(session: Session, data: EvidenceFile, now: datetime | None = None) -> LoadReport:
    """Upsert a validated file. Commits once, at the end, or not at all."""
    moment = now or datetime.now(UTC)
    report = LoadReport()
    try:
        for r in data.roles:
            upsert_role(session, r)
        session.flush()
        for p in data.projects:
            upsert_project(session, p)
        session.flush()
        for item in data.evidence:
            record, outcome, verified = upsert_evidence(session, item, moment)
            {
                Outcome.CREATED: report.created,
                Outcome.REVISED: report.revised,
                Outcome.UNCHANGED: report.unchanged,
            }[outcome].append(record.id)
            if verified:
                report.verified.append(record.id)
    except Exception:
        session.rollback()
        raise
    session.commit()
    return report


def verify(
    session: Session,
    evidence_id: str,
    method: VerificationMethod,
    by: str,
    now: datetime | None = None,
) -> Evidence:
    record = session.get(Evidence, evidence_id)
    if record is None:
        raise EvidenceError(f"no evidence '{evidence_id}'")
    if method is VerificationMethod.ARTIFACT and not record.artifact_url:
        raise EvidenceError(f"'{evidence_id}' has no artifact_url to verify against")
    record.verification_status = VerificationStatus.VERIFIED
    record.verification_method = method
    record.verified_by = by
    record.verified_at = now or datetime.now(UTC)
    session.commit()
    return record


def reject(session: Session, evidence_id: str, by: str, now: datetime | None = None) -> Evidence:
    record = session.get(Evidence, evidence_id)
    if record is None:
        raise EvidenceError(f"no evidence '{evidence_id}'")
    record.verification_status = VerificationStatus.REJECTED
    record.verification_method = None
    record.verified_by = by
    record.verified_at = now or datetime.now(UTC)
    session.commit()
    return record


def citable(session: Session) -> list[Evidence]:
    """The only records the generator may cite."""
    return list(
        session.scalars(
            select(Evidence)
            .where(Evidence.verification_status == VerificationStatus.VERIFIED)
            .order_by(Evidence.id)
        )
    )
