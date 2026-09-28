"""Retrieval ablation: which configuration actually finds the right evidence?

    python -m grounded.retrieval ablate benchmarks/retrieval/queries.yaml \\
        --corpus benchmarks/retrieval/corpus.yaml --out docs/results/retrieval-ablation.md

Loads the corpus into a throwaway in-memory store, runs every labelled query
through each configuration, and reports recall@k and MRR. Configurations
whose models are not installed are skipped and listed as skipped — never
silently substituted.

The shipped benchmark is **synthetic**: a fictional career, labelled by the
author before any configuration was run. It shows how configurations
differ; it is not the golden set (step 50), which is real and human-labelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from grounded.evidence.models import Base
from grounded.evidence.store import load, read_file
from grounded.retrieval.embedding import Embedder, get_embedder
from grounded.retrieval.index import embed_pending
from grounded.retrieval.rerank import Reranker, get_reranker
from grounded.retrieval.search import Mode, search

KS = (1, 3, 5, 10)


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    query: str
    relevant: list[str] = Field(min_length=1)


class Benchmark(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    queries: list[Query] = Field(min_length=1)


@dataclass(frozen=True)
class Config:
    name: str
    mode: Mode
    embedder: str | None = None
    reranker: str | None = None


CONFIGS = (
    Config("BM25", Mode.BM25),
    Config("Dense / hashing", Mode.DENSE, "hashing"),
    Config("Dense / bge-small", Mode.DENSE, "bge-small"),
    Config("Hybrid / hashing", Mode.HYBRID, "hashing"),
    Config("Hybrid / bge-small", Mode.HYBRID, "bge-small"),
    Config("Hybrid / bge-small + rerank", Mode.HYBRID, "bge-small", "minilm"),
)


@dataclass
class Result:
    config: Config
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    misses: dict[str, list[str]] = field(default_factory=dict)
    """Query id → relevant records not in the top 10."""


def read_benchmark(path: Path) -> Benchmark:
    return Benchmark.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def evaluate(ranked: list[list[str]], benchmark: Benchmark, config: Config) -> Result:
    result = Result(config)
    for k in KS:
        per_query = [
            len(set(r[:k]) & set(q.relevant)) / len(q.relevant)
            for r, q in zip(ranked, benchmark.queries, strict=True)
        ]
        result.recall[k] = sum(per_query) / len(per_query)
    reciprocal = []
    for r, q in zip(ranked, benchmark.queries, strict=True):
        first = next((i for i, e in enumerate(r, start=1) if e in q.relevant), None)
        reciprocal.append(1 / first if first else 0.0)
        missing = [e for e in q.relevant if e not in r[:10]]
        if missing:
            result.misses[q.id] = missing
    result.mrr = sum(reciprocal) / len(reciprocal)
    return result


def select(names: list[str] | None) -> tuple[Config, ...]:
    """Configurations by name (case-insensitive); all of them when `names` is empty."""
    if not names:
        return CONFIGS
    wanted = {n.lower() for n in names}
    chosen = tuple(c for c in CONFIGS if c.name.lower() in wanted)
    unknown = wanted - {c.name.lower() for c in chosen}
    if unknown:
        raise ValueError(
            f"unknown configuration(s): {', '.join(sorted(unknown))} — "
            f"choose from: {', '.join(c.name for c in CONFIGS)}"
        )
    return chosen


def run(
    corpus: Path, benchmark: Benchmark, configs: tuple[Config, ...] = CONFIGS
) -> tuple[list[Result], list[tuple[str, str]]]:
    """Returns results and (config, reason) for every configuration skipped."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    results: list[Result] = []
    skipped: list[tuple[str, str]] = []
    embedders: dict[str, Embedder] = {}
    rerankers: dict[str, Reranker] = {}
    with Session(engine) as session:
        load(session, read_file(corpus))
        for config in configs:
            try:
                if config.embedder and config.embedder not in embedders:
                    embedders[config.embedder] = get_embedder(config.embedder)
                    embed_pending(session, embedders[config.embedder])
                if config.reranker and config.reranker not in rerankers:
                    rerankers[config.reranker] = get_reranker(config.reranker)
            except Exception as exc:
                # Reported, never substituted: a skipped row is honest, a
                # quietly different model under the same name is not.
                reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                skipped.append((config.name, reason))
                continue
            embedder = embedders.get(config.embedder) if config.embedder else None
            reranker = rerankers.get(config.reranker) if config.reranker else None
            ranked = [
                [
                    h.evidence_id
                    for h in search(
                        session, q.query, embedder, k=10, mode=config.mode, reranker=reranker
                    )
                ]
                for q in benchmark.queries
            ]
            results.append(evaluate(ranked, benchmark, config))
    return results, skipped


def markdown(results: list[Result], skipped: list[tuple[str, str]], benchmark: Benchmark) -> str:
    header = "| Configuration | " + " | ".join(f"Recall@{k}" for k in KS) + " | MRR |"
    lines = [
        f"# Retrieval ablation — {date.today().isoformat()}",
        "",
        f"Synthetic benchmark: {len(benchmark.queries)} labelled queries over a fictional "
        "career (`benchmarks/retrieval/`), labelled by the author before any run. "
        "It compares configurations; it is not the golden set.",
        "",
        header,
        "|" + "---|" * (len(KS) + 2),
    ]
    for r in results:
        cells = " | ".join(f"{r.recall[k]:.2f}" for k in KS)
        lines.append(f"| {r.config.name} | {cells} | {r.mrr:.2f} |")
    if skipped:
        lines += ["", "Skipped (model unavailable): " + ", ".join(n for n, _ in skipped)]
    lines += ["", "## Misses at k = 10", ""]
    for r in results:
        detail = "; ".join(f"{q}: {', '.join(m)}" for q, m in r.misses.items()) or "none"
        lines.append(f"- **{r.config.name}** — {detail}")
    return "\n".join(lines) + "\n"
