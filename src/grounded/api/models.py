"""API response shapes. Requests reuse the file schema, so there is one
definition of a valid record whether it arrives as YAML or JSON."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from grounded.evidence.enums import VerificationMethod
from grounded.evidence.store import Outcome


class EvidenceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    kind: str
    statement: str
    project_id: str | None
    role_id: str | None
    month: str | None
    metric_value: Decimal | None
    metric_unit: str | None
    skills: list[str]
    artifact_url: str | None
    verification_status: str
    verification_method: str | None
    verified_by: str | None
    verified_at: datetime | None
    revision: int
    citable: bool


class WriteResult(BaseModel):
    outcome: Outcome
    verified_by_this_write: bool
    record: EvidenceOut


class RoleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    organisation: str
    start_month: str | None
    end_month: str | None


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    role_id: str | None
    summary: str | None
    repo_url: str | None


class VerifyIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: VerificationMethod
    by: str


class RejectIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    by: str
