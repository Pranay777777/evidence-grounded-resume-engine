"""A small UI for drafts (step 62): citations inline, evidence on hover, and
a base-vs-tailored diff for every kept bullet.

Server-rendered like the admin (no JavaScript build, autoescaped templates),
and like the admin it is a local tool: served only with AUTH=off (ADR-017).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, Response
from sqlalchemy.orm import Session

from grounded.api import csrf
from grounded.api.admin import templates
from grounded.api.deps import get_session
from grounded.config import get_settings
from grounded.evidence.models import Evidence
from grounded.generation.diff import Op, added_share, word_diff
from grounded.generation.generate import GenerationError
from grounded.generation.llm import LLMError
from grounded.generation.service import DraftOutcome, NoEvidenceError, create_draft

router = APIRouter(prefix="/ui")
SessionDep = Annotated[Session, Depends(get_session)]


@dataclass(frozen=True)
class Citation:
    number: int
    record: Evidence


@dataclass(frozen=True)
class KeptView:
    text: str
    citations: list[Citation]
    entailment: float | None
    base: str
    diff: list[tuple[Op, str]]
    added_share: float


def build_view(session: Session, outcome: DraftOutcome) -> dict[str, Any]:
    numbers: dict[str, int] = {}
    kept: list[KeptView] = []
    for k in outcome.report.kept:
        citations = []
        for evidence_id in k.bullet.evidence_ids:
            record = session.get(Evidence, evidence_id)
            if record is None:  # pragma: no cover - the gate only keeps resolvable citations
                continue
            numbers.setdefault(evidence_id, len(numbers) + 1)
            citations.append(Citation(numbers[evidence_id], record))
        base = " / ".join(c.record.statement for c in citations)
        diff = word_diff(base, k.bullet.text)
        kept.append(KeptView(k.bullet.text, citations, k.entailment, base, diff, added_share(diff)))
    return {
        "kept": kept,
        "dropped": outcome.report.dropped,
        "generation": outcome.generation,
        "verified": outcome.report.verified,
        "threshold": outcome.report.threshold,
        "cache_similarity": outcome.cache_similarity,
        "sources": sorted(numbers.items(), key=lambda item: item[1]),
    }


def _render(request: Request, context: dict[str, Any], status: int = 200) -> Response:
    token, new = csrf.issue(request)
    response = templates.TemplateResponse(
        request, "ui.html", {**context, "csrf_token": token}, status_code=status
    )
    if new:
        csrf.attach(response, token)
    return response


@router.get("", response_class=HTMLResponse)
def form(request: Request) -> Response:
    return _render(request, {"job_description": "", "result": None, "error": None})


@router.post("/draft", response_class=HTMLResponse)
def draft(
    request: Request,
    session: SessionDep,
    job_description: Annotated[str, Form()] = "",
    rerank: Annotated[bool, Form()] = False,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    csrf.check(request, csrf_token)
    context: dict[str, Any] = {"job_description": job_description, "result": None, "error": None}
    if len(job_description.strip()) < 20:
        context["error"] = "Paste a job description (at least 20 characters)."
        return _render(request, context, 422)
    state = request.app.state
    try:
        outcome = create_draft(
            session,
            job_description,
            get_settings(),
            state.client_factory,
            state.get_verifier(),
            rerank=rerank,
        )
    except NoEvidenceError as exc:
        context["error"] = str(exc)
        return _render(request, context, 422)
    except (LLMError, GenerationError) as exc:
        context["error"] = f"The model call failed: {exc}"
        return _render(request, context, 502)
    context["result"] = build_view(session, outcome)
    return _render(request, context)
