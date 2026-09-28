"""The evidence file format — what a person writes, validated before it is stored.

    roles:
      - id: insight-de-trainee
        title: Data Engineering Trainee
        organisation: Insight Enterprises
        start: 2026-06
    projects:
      - id: metadata-driven-lakehouse
        name: metadata-driven-lakehouse
        repo_url: https://github.com/...
    evidence:
      - id: lakehouse-test-coverage
        kind: metric
        statement: The lakehouse test suite has 499 tests at 96% line coverage.
        project: metadata-driven-lakehouse
        month: 2026-09
        metric: {value: 96, unit: percent}
        artifact_url: https://github.com/.../actions
        verification: {method: artifact, by: GitHub Actions}

Validation happens here, with messages a person can act on, so the
database constraints are a second line rather than the first.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from grounded.evidence.enums import EvidenceKind, VerificationMethod
from grounded.evidence.models import ID_PATTERN, MONTH_PATTERN

Slug = Annotated[str, Field(pattern=ID_PATTERN, max_length=80)]
Month = Annotated[str, Field(pattern=MONTH_PATTERN)]


def _month(value: object) -> object:
    """YAML reads `2026-06` as a string but `2026-06-01` as a date; accept both."""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m")
    return value


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RoleIn(_Strict):
    id: Slug
    title: str = Field(min_length=1, max_length=200)
    organisation: str = Field(min_length=1, max_length=200)
    start: Month | None = None
    end: Month | None = None

    _months = field_validator("start", "end", mode="before")(_month)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.start and self.end and self.start > self.end:
            raise ValueError(f"role '{self.id}' ends ({self.end}) before it starts ({self.start})")
        return self


class ProjectIn(_Strict):
    id: Slug
    name: str = Field(min_length=1, max_length=200)
    role: Slug | None = None
    summary: str | None = None
    repo_url: HttpUrl | None = None


class MetricIn(_Strict):
    value: Decimal
    unit: str = Field(min_length=1, max_length=40)


class VerificationIn(_Strict):
    method: VerificationMethod
    by: str = Field(min_length=1, max_length=200)


class EvidenceIn(_Strict):
    id: Slug
    kind: EvidenceKind
    statement: str = Field(min_length=10, max_length=400)
    project: Slug | None = None
    role: Slug | None = None
    month: Month | None = None
    metric: MetricIn | None = None
    skills: list[str] = Field(default_factory=list)
    artifact_url: HttpUrl | None = None
    verification: VerificationIn | None = None

    _months = field_validator("month", mode="before")(_month)

    @model_validator(mode="after")
    def _rules(self) -> Self:
        if (self.kind is EvidenceKind.METRIC) != (self.metric is not None):
            raise ValueError(
                f"'{self.id}': a metric record needs `metric`, and only metric records may have one"
            )
        if (
            self.verification is not None
            and self.verification.method is VerificationMethod.ARTIFACT
            and self.artifact_url is None
        ):
            raise ValueError(f"'{self.id}': verification by artifact needs an `artifact_url`")
        return self

    def content_hash(self) -> str:
        """Hash of the fact itself. Verification is not part of it: verifying
        a record does not change what it says."""
        fact = {
            "kind": self.kind.value,
            "statement": self.statement.strip(),
            "project": self.project,
            "role": self.role,
            "month": self.month,
            "metric": [str(self.metric.value), self.metric.unit] if self.metric else None,
            "artifact_url": str(self.artifact_url) if self.artifact_url else None,
        }
        encoded = json.dumps(fact, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class EvidenceFile(_Strict):
    roles: list[RoleIn] = Field(default_factory=list)
    projects: list[ProjectIn] = Field(default_factory=list)
    evidence: list[EvidenceIn] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        for label, items in (
            ("role", self.roles),
            ("project", self.projects),
            ("evidence", self.evidence),
        ):
            seen: set[str] = set()
            for item in items:
                if item.id in seen:
                    raise ValueError(f"duplicate {label} id '{item.id}'")
                seen.add(item.id)
        return self
