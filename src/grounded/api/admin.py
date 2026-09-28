"""A small server-rendered admin for adding, editing and verifying evidence.

Plain HTML forms, no JavaScript build. Every write goes through the same
`upsert_evidence` as the API and the loader, so editing a statement here
bumps its revision and clears its verification exactly as a file edit
would (ADR-002).
"""

from __future__ import annotations

from importlib.resources import files
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from grounded.api import csrf
from grounded.api.deps import get_session
from grounded.evidence.enums import EvidenceKind, VerificationMethod, VerificationStatus
from grounded.evidence.models import Evidence, Project, Role
from grounded.evidence.schema import EvidenceIn
from grounded.evidence.store import EvidenceError, reject, upsert_evidence, verify

router = APIRouter(prefix="/admin")
templates = Jinja2Templates(directory=str(files("grounded") / "api" / "templates"))
SessionDep = Annotated[Session, Depends(get_session)]
Text = Annotated[str, Form()]
OptionalText = Annotated[str, Form()]


def _render(request: Request, name: str, context: dict[str, Any], status: int = 200) -> Response:
    token, new = csrf.issue(request)
    response = templates.TemplateResponse(
        request, name, {**context, "csrf_token": token}, status_code=status
    )
    if new:
        csrf.attach(response, token)
    return response


def _choices(session: Session) -> dict[str, Any]:
    return {
        "kinds": [k.value for k in EvidenceKind],
        "methods": [m.value for m in VerificationMethod],
        "projects": list(session.scalars(select(Project.id).order_by(Project.id))),
        "roles": list(session.scalars(select(Role.id).order_by(Role.id))),
    }


@router.get("", response_class=HTMLResponse)
def index(request: Request, session: SessionDep, status: str | None = None) -> Response:
    counts = dict(
        session.execute(
            select(Evidence.verification_status, func.count()).group_by(
                Evidence.verification_status
            )
        ).all()
    )
    query = select(Evidence).order_by(Evidence.id)
    if status in {s.value for s in VerificationStatus}:
        query = query.where(Evidence.verification_status == status)
    return _render(
        request,
        "index.html",
        {"records": list(session.scalars(query)), "counts": counts, "status": status},
    )


@router.get("/evidence/new", response_class=HTMLResponse)
def new(request: Request, session: SessionDep) -> Response:
    return _render(request, "edit.html", {"record": None, "errors": [], **_choices(session)})


@router.get("/evidence/{evidence_id}", response_class=HTMLResponse)
def detail(evidence_id: str, request: Request, session: SessionDep) -> Response:
    record = session.get(Evidence, evidence_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no evidence '{evidence_id}'")
    return _render(request, "edit.html", {"record": record, "errors": [], **_choices(session)})


@router.post("/evidence")
def save(
    request: Request,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
    id: Text = "",  # noqa: A002 — the field is called id in every other representation
    kind: Text = "",
    statement: Text = "",
    project: OptionalText = "",
    role: OptionalText = "",
    month: OptionalText = "",
    metric_value: OptionalText = "",
    metric_unit: OptionalText = "",
    skills: OptionalText = "",
    artifact_url: OptionalText = "",
) -> Response:
    csrf.check(request, csrf_token)
    raw: dict[str, Any] = {
        "id": id.strip(),
        "kind": kind,
        "statement": statement,
        "project": project or None,
        "role": role or None,
        "month": month or None,
        "metric": {"value": metric_value, "unit": metric_unit} if metric_value else None,
        "skills": [s.strip() for s in skills.split(",") if s.strip()],
        "artifact_url": artifact_url or None,
    }
    try:
        item = EvidenceIn.model_validate(raw)
        record, _, _ = upsert_evidence(session, item)
        session.commit()
    except (ValidationError, EvidenceError) as exc:
        session.rollback()
        errors = (
            [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
            if isinstance(exc, ValidationError)
            else [str(exc)]
        )
        return _render(
            request,
            "edit.html",
            {"record": None, "form": raw, "errors": errors, **_choices(session)},
            status=400,
        )
    return RedirectResponse(f"/admin/evidence/{record.id}", status_code=303)


@router.post("/evidence/{evidence_id}/verify")
def verify_form(
    evidence_id: str,
    request: Request,
    session: SessionDep,
    method: Text,
    by: Text,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    csrf.check(request, csrf_token)
    try:
        verify(session, evidence_id, VerificationMethod(method), by)
    except (EvidenceError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RedirectResponse(f"/admin/evidence/{evidence_id}", status_code=303)


@router.post("/evidence/{evidence_id}/reject")
def reject_form(
    evidence_id: str,
    request: Request,
    session: SessionDep,
    by: Text,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    csrf.check(request, csrf_token)
    try:
        reject(session, evidence_id, by)
    except EvidenceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RedirectResponse(f"/admin/evidence/{evidence_id}", status_code=303)
