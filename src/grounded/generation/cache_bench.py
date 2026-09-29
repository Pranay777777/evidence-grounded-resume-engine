"""Measure the semantic cache without calling a model (ADR-013).

    python -m grounded.generation cache-bench benchmarks/cache/pairs.yaml \\
        --out docs/results/semantic-cache.md

Each pair is two job descriptions labelled `same` (a reformatted or
paraphrased version of one request - a hit is correct) or not (a different
request - a hit would serve the wrong draft). For each threshold it reports
the hit rate on `same` pairs and the false-hit rate on the rest, twice:
on embedding similarity alone, and with the cache's real rule, which also
requires both requests to retrieve the identical evidence set.

Tokens and dollars saved are projected from the hit rate and the tokens a
draft actually used (recorded by `evals collect` in runs.jsonl).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from grounded.evidence.models import Base
from grounded.evidence.store import load, read_file
from grounded.generation.cache import cosine, normalise
from grounded.retrieval.embedding import Embedder
from grounded.retrieval.index import embed_pending
from grounded.retrieval.search import search

THRESHOLDS = (0.80, 0.85, 0.90, 0.95, 0.97, 0.99)


class CachePair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    kind: str
    """What varies: format, paraphrase, stack, role... (for reading the report)."""
    same: bool
    a: str
    b: str


class CachePairSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pairs: list[CachePair] = Field(min_length=1)


@dataclass(frozen=True)
class Scored:
    pair: CachePair
    similarity: float
    same_evidence: bool


@dataclass(frozen=True)
class BenchRow:
    threshold: float
    hit_rate: float
    false_hits: float
    hit_rate_with_evidence: float
    false_hits_with_evidence: float


def read_pairs(path: Path) -> CachePairSet:
    return CachePairSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def score(pairs: CachePairSet, corpus: Path, embedder: Embedder, k: int = 12) -> list[Scored]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    scored = []
    with Session(engine) as session:
        load(session, read_file(corpus))
        embed_pending(session, embedder)
        for pair in pairs.pairs:
            if normalise(pair.a) == normalise(pair.b):
                similarity = 1.0
            else:
                va, vb = embedder.embed([pair.a, pair.b])
                similarity = cosine(va, vb)
            ids_a = {h.evidence_id for h in search(session, pair.a, embedder, k=k)}
            ids_b = {h.evidence_id for h in search(session, pair.b, embedder, k=k)}
            scored.append(Scored(pair, similarity, ids_a == ids_b))
    return scored


def _rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def sweep(scored: list[Scored]) -> list[BenchRow]:
    same = [s for s in scored if s.pair.same]
    different = [s for s in scored if not s.pair.same]
    rows = []
    for t in THRESHOLDS:
        rows.append(
            BenchRow(
                threshold=t,
                hit_rate=_rate(sum(s.similarity >= t for s in same), len(same)),
                false_hits=_rate(sum(s.similarity >= t for s in different), len(different)),
                hit_rate_with_evidence=_rate(
                    sum(s.similarity >= t and s.same_evidence for s in same), len(same)
                ),
                false_hits_with_evidence=_rate(
                    sum(s.similarity >= t and s.same_evidence for s in different), len(different)
                ),
            )
        )
    return rows


def markdown(
    scored: list[Scored],
    embedder_name: str,
    threshold: float,
    tokens_per_draft: float | None,
    price_per_mtok: float,
) -> str:
    same = sum(s.pair.same for s in scored)
    lines = [
        f"# Semantic cache benchmark - {date.today().isoformat()}",
        "",
        f"{len(scored)} labelled job-description pairs ({same} the same request, "
        f"{len(scored) - same} different) from `benchmarks/cache/pairs.yaml`; embedder "
        f"`{embedder_name}`. No model was called.",
        "",
        "| Threshold | Hit rate (similarity) | False hits (similarity) "
        "| Hit rate (+ same evidence) | False hits (+ same evidence) |",
        "|---|---|---|---|---|",
    ]
    rows = sweep(scored)
    for r in rows:
        mark = " (configured)" if abs(r.threshold - threshold) < 1e-9 else ""
        lines.append(
            f"| {r.threshold:.2f}{mark} | {r.hit_rate:.0%} | {r.false_hits:.0%} | "
            f"{r.hit_rate_with_evidence:.0%} | {r.false_hits_with_evidence:.0%} |"
        )
    lines += ["", "The cache's rule is the right-hand pair of columns.", ""]
    configured = next((r for r in rows if abs(r.threshold - threshold) < 1e-9), None)
    if configured is None:
        lines.append(f"The configured threshold {threshold} is not in the sweep.")
    elif tokens_per_draft is None:
        lines.append(
            "Savings: no token counts recorded yet - run `python -m grounded.evals collect` "
            "with a new model to record them in runs.jsonl."
        )
    else:
        per_100 = configured.hit_rate_with_evidence * 100 * tokens_per_draft
        dollars = per_100 / 1_000_000 * price_per_mtok
        lines.append(
            f"Savings at {threshold}: if every request were a repeat or rephrase, "
            f"{configured.hit_rate_with_evidence:.0%} of them hit, saving about "
            f"{per_100:,.0f} tokens per 100 requests (mean {tokens_per_draft:,.0f} tokens per "
            f"draft) = ${dollars:.4f} at ${price_per_mtok}/M tokens"
            + (" (free models cost $0)." if price_per_mtok == 0 else ".")
        )
    lines += [
        "",
        "## Per pair",
        "",
        "| Pair | Kind | Same request | Similarity | Same evidence |",
        "|---|---|---|---|---|",
    ]
    for s in scored:
        lines.append(
            f"| {s.pair.id} | {s.pair.kind} | {'yes' if s.pair.same else 'no'} | "
            f"{s.similarity:.3f} | {'yes' if s.same_evidence else 'no'} |"
        )
    return "\n".join(lines) + "\n"
