"""One draft, end to end: retrieve, (reuse or) generate, restore, gate.

The CLI and the HTTP API both call `create_draft`, so they cannot drift
apart: the same retrieval, prompt registry, redaction, cache and grounding
gate run behind both.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from grounded.config import Settings
from grounded.evidence.models import Evidence, Project
from grounded.generation import prompt
from grounded.generation.adapters import is_local
from grounded.generation.cache import SemanticCache, request_key
from grounded.generation.generate import ChatClient, Generation, generate
from grounded.generation.privacy import Redaction, get_finder
from grounded.retrieval.embedding import get_embedder
from grounded.retrieval.rerank import get_reranker
from grounded.retrieval.search import search
from grounded.verification.grounding import GroundingReport, ground
from grounded.verification.nli import Verifier

ClientFactory = Callable[[str, Settings], ChatClient]


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
    records = retrieve(session, job, settings, k, rerank)

    cache = key = None
    hit = None
    if use_cache:
        cache = SemanticCache(
            settings.semantic_cache_path_obj,
            get_embedder(settings.embedder),
            settings.cache_threshold,
        )
        key = request_key(settings.llm_model, fingerprint, [(r.id, r.revision) for r in records])
        hit = cache.lookup(key, job)

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
                settings.redaction, [t for t in settings.redact_terms.split(",") if t.strip()]
            )
        )
        client = client_factory(settings.llm_model, settings)
        generation = generate(
            client,
            job,
            records,
            prompt_version=version,
            redaction=Redaction(finder) if finder else None,
        )
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

    candidates = {r.id: r.revision for r in records}
    report = ground(session, generation.draft, candidates, verifier, settings.verifier_threshold)
    return DraftOutcome(generation, report, hit.similarity if hit else None)
