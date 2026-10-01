"""A small UI for drafts (step 62): citations inline, evidence on hover, and
a base-vs-tailored diff for every kept bullet.

Server-rendered like the admin (no JavaScript build, autoescaped templates),
and like the admin it is a local tool: served only with AUTH=off (ADR-017).

**Samples** replay a recorded generation from the golden set through the live
gate - the bullets a real model wrote for that job description, judged now,
with no model call. In the public demo (ADR-020) they are what visitors see
first; drafting from your own text is rate limited per visitor and bounded by
a daily token budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, Response
from sqlalchemy.orm import Session

from grounded.api import csrf
from grounded.api.admin import templates
from grounded.api.deps import get_session
from grounded.api.drafts import limiter
from grounded.config import get_settings
from grounded.evals.golden import GoldenItem, Job, read_items, read_jobs
from grounded.evidence.models import Evidence
from grounded.generation.diff import Op, added_share, word_diff
from grounded.generation.generate import Generation, GenerationError
from grounded.generation.llm import LLMError
from grounded.generation.schema import Bullet, Draft
from grounded.generation.service import DraftOutcome, NoEvidenceError, create_draft
from grounded.verification.grounding import ground

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


@dataclass(frozen=True)
class Sample:
    job: Job
    model: str
    items: list[GoldenItem]


@lru_cache(maxsize=4)
def _samples(golden: str, jobs: str) -> tuple[Sample, ...]:
    """One recorded draft per job: the model with the most bullets for it."""
    if not Path(golden).exists() or not Path(jobs).exists():
        return ()
    items = read_items(Path(golden))
    out = []
    for job in read_jobs(Path(jobs)).jds:
        mine = [i for i in items if i.jd_id == job.id]
        if not mine:
            continue
        models = sorted(
            {i.model for i in mine}, key=lambda m: (-sum(i.model == m for i in mine), m)
        )
        out.append(Sample(job, models[0], [i for i in mine if i.model == models[0]]))
    return tuple(out)


def samples() -> tuple[Sample, ...]:
    settings = get_settings()
    return _samples(settings.golden_path, settings.jobs_path)


def _demo_exempt() -> bool:
    return not get_settings().demo


def _demo_rate() -> str:
    return get_settings().demo_rate


DEMO_BUDGET = "demo"


@dataclass(frozen=True)
class _Budget:
    name: str
    daily_tokens: int


def _render(request: Request, context: dict[str, Any], status: int = 200) -> Response:
    token, new = csrf.issue(request)
    settings = get_settings()
    context = {
        **context,
        "csrf_token": token,
        "samples": samples(),
        "demo": settings.demo,
        "demo_rate": settings.demo_rate,
    }
    response = templates.TemplateResponse(request, "ui.html", context, status_code=status)
    if new:
        csrf.attach(response, token)
    return response


@router.get("", response_class=HTMLResponse)
def form(request: Request) -> Response:
    return _render(request, {"job_description": "", "result": None, "error": None})


@router.post("/draft", response_class=HTMLResponse)
@limiter.limit(_demo_rate, exempt_when=_demo_exempt)
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
    settings = get_settings()
    budget = _Budget(DEMO_BUDGET, settings.demo_daily_tokens)
    if settings.demo and state.ledger.remaining(budget) <= 0:
        context["error"] = (
            "Today's model budget for the demo is used up - try one of the samples, "
            "which replay recorded drafts through the live gate."
        )
        return _render(request, context, 429)
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
    if settings.demo:
        state.ledger.charge(budget, outcome.tokens)
    context["result"] = build_view(session, outcome)
    return _render(request, context)


@router.post("/sample", response_class=HTMLResponse)
def sample(
    request: Request,
    session: SessionDep,
    jd_id: Annotated[str, Form()] = "",
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """A recorded generation, judged by the live gate. No model call."""
    csrf.check(request, csrf_token)
    chosen = next((s for s in samples() if s.job.id == jd_id), None)
    if chosen is None:
        return _render(
            request,
            {"job_description": "", "result": None, "error": f"unknown sample '{jd_id}'"},
            404,
        )
    draft = Draft(
        bullets=[
            Bullet(text=i.text, evidence_ids=i.evidence_ids or ["-"]) for i in chosen.items[:8]
        ]
    )
    cited = {i for item in chosen.items for i in item.evidence_ids}
    records = [r for r in (session.get(Evidence, i) for i in cited) if r is not None]
    report = ground(
        session,
        draft,
        {r.id: r.revision for r in records},
        request.app.state.get_verifier(),
        get_settings().verifier_threshold,
    )
    generation = Generation(
        draft=draft, attempts=0, model=chosen.model, prompt_version=chosen.items[0].prompt_version
    )
    view = build_view(session, DraftOutcome(generation, report, None))
    view["sample"] = chosen
    return _render(request, {"job_description": chosen.job.text, "result": view, "error": None})
