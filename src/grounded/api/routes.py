"""JSON API over the evidence store.

Writes go through `store.upsert_evidence`, the same function the YAML loader
uses, so ADR-002's rules — an edit is a new revision, and a new revision is
unverified — cannot be bypassed by choosing the API over the file. There are
no DELETE routes: a record is retired by rejecting it, which keeps it on the
record (ADR-002).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from grounded.api.deps import get_session
from grounded.api.models import (
    EvidenceOut,
    ProjectOut,
    RejectIn,
    RoleOut,
    VerifyIn,
    WriteResult,
)
from grounded.evidence.enums import EvidenceKind, VerificationStatus
from grounded.evidence.models import Evidence, Project, Role
from grounded.evidence.schema import EvidenceIn, ProjectIn, RoleIn
from grounded.evidence.store import (
    EvidenceError,
    reject,
    upsert_evidence,
    upsert_project,
    upsert_role,
    verify,
)

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


def _found(session: Session, model: type[Evidence | Role | Project], key: str) -> None:
    if session.get(model, key) is None:
        raise HTTPException(status_code=404, detail=f"no {model.__tablename__} '{key}'")


def _same_id(path_id: str, body_id: str) -> None:
    if path_id != body_id:
        raise HTTPException(
            status_code=422, detail=f"body id '{body_id}' does not match path id '{path_id}'"
        )


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/evidence", response_model=list[EvidenceOut])
def list_evidence(
    session: SessionDep,
    status: VerificationStatus | None = None,
    kind: EvidenceKind | None = None,
    project: str | None = None,
) -> list[Evidence]:
    query = select(Evidence).order_by(Evidence.id)
    if status is not None:
        query = query.where(Evidence.verification_status == status)
    if kind is not None:
        query = query.where(Evidence.kind == kind)
    if project is not None:
        query = query.where(Evidence.project_id == project)
    return list(session.scalars(query))


@router.get("/evidence/{evidence_id}", response_model=EvidenceOut)
def get_evidence(evidence_id: str, session: SessionDep) -> Evidence:
    record = session.get(Evidence, evidence_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no evidence '{evidence_id}'")
    return record


@router.put("/evidence/{evidence_id}", response_model=WriteResult)
def put_evidence(evidence_id: str, body: EvidenceIn, session: SessionDep) -> WriteResult:
    """Create or replace a record. A changed fact comes back unverified."""
    _same_id(evidence_id, body.id)
    try:
        record, outcome, verified = upsert_evidence(session, body)
    except EvidenceError as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    session.commit()
    return WriteResult(
        outcome=outcome,
        verified_by_this_write=verified,
        record=EvidenceOut.model_validate(record),
    )


@router.post("/evidence/{evidence_id}/verify", response_model=EvidenceOut)
def verify_evidence(evidence_id: str, body: VerifyIn, session: SessionDep) -> Evidence:
    _found(session, Evidence, evidence_id)
    try:
        return verify(session, evidence_id, body.method, body.by)
    except EvidenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/evidence/{evidence_id}/reject", response_model=EvidenceOut)
def reject_evidence(evidence_id: str, body: RejectIn, session: SessionDep) -> Evidence:
    _found(session, Evidence, evidence_id)
    return reject(session, evidence_id, body.by)


@router.get("/roles", response_model=list[RoleOut])
def list_roles(session: SessionDep) -> list[Role]:
    return list(session.scalars(select(Role).order_by(Role.id)))


@router.put("/roles/{role_id}", response_model=RoleOut)
def put_role(role_id: str, body: RoleIn, session: SessionDep) -> Role:
    _same_id(role_id, body.id)
    role = upsert_role(session, body)
    session.commit()
    return role


@router.get("/projects", response_model=list[ProjectOut])
def list_projects(session: SessionDep) -> list[Project]:
    return list(session.scalars(select(Project).order_by(Project.id)))


@router.put("/projects/{project_id}", response_model=ProjectOut)
def put_project(project_id: str, body: ProjectIn, session: SessionDep) -> Project:
    _same_id(project_id, body.id)
    try:
        project = upsert_project(session, body)
    except EvidenceError as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    session.commit()
    return project
