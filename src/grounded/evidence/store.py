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
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from grounded.evidence.enums import VerificationMethod, VerificationStatus
from grounded.evidence.models import Evidence, Project, Role
from grounded.evidence.schema import EvidenceFile, EvidenceIn


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


def _check_references(session: Session, data: EvidenceFile) -> None:
    roles = {r.id for r in data.roles} | set(session.scalars(select(Role.id)))
    projects = {p.id for p in data.projects} | set(session.scalars(select(Project.id)))
    for p in data.projects:
        if p.role and p.role not in roles:
            raise EvidenceError(f"project '{p.id}' refers to unknown role '{p.role}'")
    for e in data.evidence:
        if e.project and e.project not in projects:
            raise EvidenceError(f"evidence '{e.id}' refers to unknown project '{e.project}'")
        if e.role and e.role not in roles:
            raise EvidenceError(f"evidence '{e.id}' refers to unknown role '{e.role}'")


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


def load(session: Session, data: EvidenceFile, now: datetime | None = None) -> LoadReport:
    """Upsert a validated file. Commits once, at the end, or not at all."""
    moment = now or datetime.now(UTC)
    _check_references(session, data)
    report = LoadReport()

    for r in data.roles:
        role = session.get(Role, r.id) or Role(id=r.id)
        role.title, role.organisation = r.title, r.organisation
        role.start_month, role.end_month = r.start, r.end
        session.add(role)
    for p in data.projects:
        project = session.get(Project, p.id) or Project(id=p.id)
        project.name, project.role_id, project.summary = p.name, p.role, p.summary
        project.repo_url = str(p.repo_url) if p.repo_url else None
        session.add(project)
    session.flush()

    for item in data.evidence:
        record = session.get(Evidence, item.id)
        if record is None:
            record = Evidence(id=item.id, revision=1)
            _write_fact(record, item)
            record.verification_status = VerificationStatus.UNVERIFIED
            session.add(record)
            report.created.append(item.id)
            if _apply_verification(record, item, moment):
                report.verified.append(item.id)
        elif record.content_hash != item.content_hash():
            _write_fact(record, item)
            record.revision += 1
            # The fact changed: whatever verified the old one does not
            # verify this one. The file's own verification block may
            # re-verify it, and that is recorded as a fresh verification.
            record.verification_status = VerificationStatus.UNVERIFIED
            record.verification_method = record.verified_by = None
            record.verified_at = None
            report.revised.append(item.id)
            if _apply_verification(record, item, moment):
                report.verified.append(item.id)
        else:
            record.skills = list(item.skills)  # tags are not part of the fact
            report.unchanged.append(item.id)
            if not record.citable and _apply_verification(record, item, moment):
                report.verified.append(item.id)

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
