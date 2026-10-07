"""Turn a profile (which records go where) and the evidence store into a résumé.

No model is involved: each bullet is the exact statement of a verified record, so the
résumé can claim nothing the store does not. The profile only chooses and orders; a
selected record that is missing, unverified or rejected stops the build.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import AnyUrl, BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from grounded.evidence.models import Evidence, Project, Role


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Link(_Strict):
    label: str = Field(min_length=1, max_length=120)
    url: AnyUrl | None = None  # https:, mailto:, tel:


class SkillGroup(_Strict):
    group: str = Field(min_length=1, max_length=60)
    items: list[str] = Field(min_length=1)


class RoleSection(_Strict):
    role: str
    evidence: list[str] = Field(min_length=1)


class ProjectSection(_Strict):
    project: str
    evidence: list[str] = Field(min_length=1)
    demo: str | None = None
    """A record verified by its URL, linked from the header as the live demo."""


class Profile(_Strict):
    """Header details and the selection. Private: it holds contact details."""

    name: str = Field(min_length=1, max_length=120)
    headline: str = Field(min_length=1, max_length=200)
    contact: list[Link] = Field(default_factory=list)
    experience: list[RoleSection] = Field(default_factory=list)
    projects: list[ProjectSection] = Field(default_factory=list)
    skills: list[SkillGroup] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


def read_profile(path: Path) -> Profile:
    return Profile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class ResumeError(ValueError):
    """The profile asks for something the store cannot back."""


@dataclass(frozen=True)
class Bullet:
    text: str
    evidence_id: str
    revision: int
    link: str | None  # the artifact, else the project's repository
    self_attested: bool


@dataclass(frozen=True)
class Block:
    title: str
    subtitle: str | None
    link: str | None
    bullets: list[Bullet]
    demo: Bullet | None = None


@dataclass(frozen=True)
class Resume:
    profile: Profile
    experience: list[Block] = field(default_factory=list)
    projects: list[Block] = field(default_factory=list)
    education: list[Bullet] = field(default_factory=list)
    certifications: list[Bullet] = field(default_factory=list)

    @property
    def cited(self) -> list[Bullet]:
        """Every record the résumé rests on, once each: bullets, then demos not already cited."""
        blocks = self.experience + self.projects
        found = [b for blk in blocks for b in blk.bullets] + self.education + self.certifications
        ids = {b.evidence_id for b in found}
        return found + [blk.demo for blk in blocks if blk.demo and blk.demo.evidence_id not in ids]


_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _month(value: str) -> str:
    """'2026-04' -> 'Apr 2026' (fixed English names: no locale, so the output is stable)."""
    year, month = value.split("-")
    return f"{_MONTHS[int(month) - 1]} {year}"


def _months(role: Role) -> str:
    end = _month(role.end_month) if role.end_month else "present"
    return f"{_month(role.start_month)} \u2013 {end}" if role.start_month else end  # en dash


def build(session: Session, profile: Profile) -> Resume:
    problems: list[str] = []
    used: set[str] = set()

    def citable(evidence_id: str) -> Evidence | None:
        record = session.get(Evidence, evidence_id)
        if record is None:
            problems.append(f"{evidence_id}: not in the evidence store")
            return None
        if not record.citable:
            problems.append(f"{evidence_id}: {record.verification_status}, not verified")
            return None
        return record

    def bullet(evidence_id: str, compact: bool = False) -> Bullet | None:
        if evidence_id in used:
            problems.append(f"{evidence_id}: selected twice")
            return None
        used.add(evidence_id)
        record = citable(evidence_id)
        if record is None:
            return None
        repo = record.project.repo_url if record.project else None
        text = record.statement
        if compact and record.metric_unit and record.metric_value is not None:
            text = f"{record.metric_unit} {record.metric_value.normalize():f}"  # "CGPA 8.3"
        return Bullet(
            text=text,
            evidence_id=record.id,
            revision=record.revision,
            link=record.artifact_url or repo,
            self_attested=record.verification_method == "self_attested",
        )

    def bullets(ids: list[str], compact: bool = False) -> list[Bullet]:
        return [b for b in (bullet(i, compact) for i in ids) if b is not None]

    def demo(evidence_id: str | None) -> Bullet | None:
        """The demo link comes from a record verified by that very URL - never typed in."""
        if evidence_id is None:
            return None
        record = citable(evidence_id)
        if record is None:
            return None
        if record.verification_method != "artifact" or not record.artifact_url:
            problems.append(f"{evidence_id}: a demo needs a record verified by its URL")
            return None
        return Bullet(record.statement, record.id, record.revision, record.artifact_url, False)

    experience = []
    for role_sec in profile.experience:
        role = session.get(Role, role_sec.role)
        if role is None:
            problems.append(f"role {role_sec.role}: not in the evidence store")
            continue
        experience.append(
            Block(
                f"{role.title} · {role.organisation}",
                _months(role),
                None,
                bullets(role_sec.evidence),
            )
        )
    projects = []
    for project_sec in profile.projects:
        project = session.get(Project, project_sec.project)
        if project is None:
            problems.append(f"project {project_sec.project}: not in the evidence store")
            continue
        projects.append(
            Block(
                project.name,
                project.summary,
                project.repo_url,
                bullets(project_sec.evidence),
                demo(project_sec.demo),
            )
        )
    # education reads as one line: a metric record shows as its unit and value
    education = bullets(profile.education, compact=True)
    certifications = bullets(profile.certifications)
    if problems:
        raise ResumeError("cannot build the résumé:\n  " + "\n  ".join(problems))
    return Resume(profile, experience, projects, education, certifications)
