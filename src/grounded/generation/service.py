"""One draft, end to end: retrieve, (reuse or) generate, restore, gate.

The CLI and the HTTP API both call `create_draft`, so they cannot drift
apart: the same retrieval, prompt registry, redaction, cache and grounding
gate run behind both.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from grounded.config import Settings
from grounded.evidence.models import Evidence, Project
from grounded.evidence.tenancy import tenant_of
from grounded.generation import prompt
from grounded.generation.adapters import is_local
from grounded.generation.cache import SemanticCache, request_key
from grounded.generation.generate import ChatClient, Generation, generate
from grounded.generation.privacy import Redaction, get_finder
from grounded.observability import cost_usd, retrieval_version, tracer
from grounded.retrieval.embedding import get_embedder
from grounded.retrieval.rerank import get_reranker
from grounded.retrieval.search import search
from grounded.verification.grounding import GroundingReport, ground
from grounded.verification.nli import Verifier

ClientFactory = Callable[[str, Settings], ChatClient]


class NoEvidenceError(ValueError):
    """Nothing verified matches - there is nothing to cite, so no model is called."""


@dataclass(frozen=True)
class DraftOutcome:
    generation: Generation
    report: GroundingReport
    cache_similarity: float | None
    """Set when the draft came from the semantic cache (ADR-013)."""

    @property
    def tokens(self) -> int:
        usage = self.generation.usage
        return usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)


def retrieve(
    session: Session, job: str, settings: Settings, k: int, rerank: bool
) -> list[Evidence]:
    reranker = get_reranker("minilm") if rerank else None
    hits = search(session, job, get_embedder(settings.embedder), k=k, reranker=reranker)
    ids = [h.evidence_id for h in hits]
    loaded = session.scalars(
        select(Evidence)
        .where(Evidence.id.in_(ids))
        .options(
            selectinload(Evidence.project).selectinload(Project.role),
            selectinload(Evidence.role),
        )
    ).all()
    by_id = {r.id: r for r in loaded}
    return [by_id[i] for i in ids if i in by_id]


def create_draft(
    session: Session,
    job: str,
    settings: Settings,
    client_factory: ClientFactory,
    verifier: Verifier | None,
    *,
    k: int = 12,
    rerank: bool = False,
    prompt_version: str | None = None,
    use_cache: bool = False,
) -> DraftOutcome:
    version = prompt_version or settings.prompt_version
    fingerprint = prompt.get(version).fingerprint  # unknown version: ValueError, early
    with tracer.start_as_current_span("draft") as span:
        span.set_attributes(
            {
                "grounded.tenant": tenant_of(session) or "-",
                "grounded.prompt.version": version,
                "grounded.prompt.fingerprint": fingerprint,
                "grounded.retrieval.version": retrieval_version(settings, k, rerank),
                "gen_ai.request.model": settings.llm_model,
            }
        )
        with tracer.start_as_current_span("retrieve") as stage:
            records = retrieve(session, job, settings, k, rerank)
            stage.set_attributes(
                {"grounded.k": k, "grounded.rerank": rerank, "grounded.hits": len(records)}
            )
        if not records:
            span.set_attribute("grounded.outcome", "no_evidence")
            raise NoEvidenceError(
                "no verified evidence matches this job description - add or verify evidence first"
            )

        cache = key = None
        hit = None
        if use_cache:
            with tracer.start_as_current_span("cache.lookup") as stage:
                cache = SemanticCache(
                    settings.semantic_cache_path_obj,
                    get_embedder(settings.embedder),
                    settings.cache_threshold,
                )
                key = request_key(
                    settings.llm_model, fingerprint, [(r.id, r.revision) for r in records]
                )
                hit = cache.lookup(key, job)
                stage.set_attribute("grounded.cache.hit", hit is not None)
                if hit is not None:
                    stage.set_attribute("grounded.cache.similarity", hit.similarity)

        if hit is not None:
            generation = Generation(
                draft=hit.entry.draft,
                attempts=0,
                model=hit.entry.model,
                prompt_version=hit.entry.prompt_version,
            )
        else:
            finder = (
                None
                if is_local(settings.llm_model)
                else get_finder(
                    settings.redaction,
                    [t for t in settings.redact_terms.split(",") if t.strip()],
                )
            )
            with tracer.start_as_current_span("llm.generate") as stage:
                stage.set_attributes(
                    {"gen_ai.operation.name": "chat", "gen_ai.request.model": settings.llm_model}
                )
                client = client_factory(settings.llm_model, settings)
                generation = generate(
                    client,
                    job,
                    records,
                    prompt_version=version,
                    redaction=Redaction(finder) if finder else None,
                )
                stage.set_attributes(_generation_attributes(generation, settings))
            if cache is not None and key is not None:
                usage = generation.usage
                cache.store(
                    key,
                    job,
                    generation.draft,
                    generation.model,
                    version,
                    usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
                )

        with tracer.start_as_current_span("gate") as stage:
            candidates = {r.id: r.revision for r in records}
            report = ground(
                session, generation.draft, candidates, verifier, settings.verifier_threshold
            )
            signal = _eval_signal(report)
            stage.set_attributes(signal)
        span.set_attributes(
            {
                **_generation_attributes(generation, settings),
                **signal,
                "grounded.cache.hit": hit is not None,
                "grounded.outcome": "ok",
            }
        )
        return DraftOutcome(generation, report, hit.similarity if hit else None)


def _generation_attributes(generation: Generation, settings: Settings) -> dict[str, Any]:
    usage = generation.usage
    attributes: dict[str, Any] = {
        "gen_ai.response.model": generation.model,
        "gen_ai.usage.input_tokens": usage.get("prompt_tokens", 0),
        "gen_ai.usage.output_tokens": usage.get("completion_tokens", 0),
        "grounded.attempts": generation.attempts,
        "grounded.retries": max(generation.attempts - 1, 0),
        "grounded.tool_calls": generation.attempts,
        "grounded.redacted": generation.redacted,
    }
    cost = cost_usd(generation.model, usage, settings) if generation.attempts else 0.0
    if cost is not None:
        attributes["grounded.cost_usd"] = cost
    return attributes


def _eval_signal(report: GroundingReport) -> dict[str, Any]:
    total = len(report.kept) + len(report.dropped)
    entailments = [k.entailment for k in report.kept if k.entailment is not None]
    reasons: dict[str, int] = {}
    for d in report.dropped:
        kind = _reason_kind(d.reason)
        reasons[kind] = reasons.get(kind, 0) + 1
    signal: dict[str, Any] = {
        "grounded.gate.kept": len(report.kept),
        "grounded.gate.dropped": len(report.dropped),
        "grounded.gate.verified": report.verified,
        "grounded.eval.kept_ratio": len(report.kept) / total if total else 0.0,
        "grounded.gate.drop_reasons": [f"{k}={v}" for k, v in sorted(reasons.items())],
    }
    if entailments:
        signal["grounded.eval.mean_entailment"] = sum(entailments) / len(entailments)
    return signal


_REASON_KINDS = (
    ("contains ", "markup"),
    ("states ", "number"),
    ("claims ", "claim_strength"),
    ("contradicted", "contradicted"),
    ("not entailed", "not_entailed"),
)


def _reason_kind(reason: str) -> str:
    """A fixed category per drop reason - counts stay comparable across drafts,
    and no bullet text ends up in a trace."""
    for prefix, kind in _REASON_KINDS:
        if reason.startswith(prefix):
            return kind
    return "citation"
