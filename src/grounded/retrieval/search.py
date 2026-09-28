"""Hybrid retrieval: BM25 and dense, fused by reciprocal rank (ADR-005).

The two retrievers fail differently, which is the whole point of using both.
BM25 finds exact terms — "Delta Lake", "HNSW", "pytest" — that an embedding
blurs into their neighbours. The dense model finds meaning — "data platform"
for a record about a lakehouse — that shares no words with the query.
Reciprocal rank fusion combines them by rank, not score, so neither
retriever's score scale has to be trusted or tuned against the other.

Only citable records at their current revision are searchable. The query is
a job description — untrusted input — and is used only as search text:
tokenised for BM25, embedded for dense search, never interpreted.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from sqlalchemy import Float, literal, select
from sqlalchemy.orm import Session

from grounded.evidence.models import EMBEDDING_DIM, Evidence, EvidenceEmbedding
from grounded.retrieval.chunking import document_text
from grounded.retrieval.embedding import Embedder, tokens
from grounded.retrieval.index import citable_records

MAX_QUERY_CHARS = 20_000
"""A job description is rarely a tenth of this. The cap stops an oversized
input from turning retrieval into a denial of service."""

RRF_K = 60
"""The constant from Cormack et al. (2009); it damps the influence of any
single retriever's top ranks. Tuned values rarely beat it by much."""

STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
        "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was",
        "were", "will", "with", "you", "your", "we", "our", "they", "their",
    }
)  # fmt: skip


class Mode(StrEnum):
    HYBRID = "hybrid"
    BM25 = "bm25"
    DENSE = "dense"


@dataclass(frozen=True)
class Hit:
    evidence_id: str
    statement: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)
    """Where each retriever placed this record — the raw material for the
    ablation in step 46."""


def query_terms(text: str) -> list[str]:
    return [t for t in tokens(text[:MAX_QUERY_CHARS]) if t not in STOPWORDS]


class BM25:
    """Okapi BM25 over a fixed corpus. Small enough to hold in memory."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in documents]
        self.lengths = [len(d) for d in documents]
        self.avg = (sum(self.lengths) / len(documents)) if documents else 0.0
        frequency: Counter[str] = Counter()
        for doc in self.docs:
            frequency.update(doc.keys())
        n = len(documents)
        # The "+1" keeps the IDF of a term that appears everywhere positive.
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in frequency.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg) if self.avg else self.k1
            total = 0.0
            for term in query:
                tf = doc.get(term, 0)
                if tf:
                    total += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            out.append(total)
        return out


def _ranked(ids: list[str], scores: list[float]) -> list[str]:
    """IDs with a positive score, best first; ties broken by ID for stability."""
    pairs = [(s, i) for i, s in zip(ids, scores, strict=True) if s > 0]
    return [i for _, i in sorted(pairs, key=lambda p: (-p[0], p[1]))]


def _dense_ranking(
    session: Session, records: list[Evidence], query: str, embedder: Embedder, depth: int
) -> list[str]:
    current = {r.id: r.revision for r in records}
    vector = embedder.embed([query[:MAX_QUERY_CHARS]])[0]
    if session.get_bind().dialect.name == "postgresql":
        from pgvector.sqlalchemy import Vector

        # `<=>` is pgvector's cosine distance, the operator the HNSW index
        # was built for. Spelled out rather than via pgvector's comparator,
        # which the cross-dialect column type does not expose.
        distance = EvidenceEmbedding.embedding.op("<=>", return_type=Float())(
            literal(vector, type_=Vector(EMBEDDING_DIM))
        )
        rows = session.execute(
            select(EvidenceEmbedding.evidence_id, EvidenceEmbedding.revision)
            .where(EvidenceEmbedding.embedder == embedder.name)
            .order_by(distance, EvidenceEmbedding.evidence_id)
            .limit(depth * 4)
        ).all()
        return [i for i, rev in rows if current.get(i) == rev][:depth]

    # SQLite (tests): no vector index, so score every current embedding.
    stored = session.scalars(
        select(EvidenceEmbedding).where(EvidenceEmbedding.embedder == embedder.name)
    ).all()
    scored = [
        (sum(a * b for a, b in zip(row.embedding, vector, strict=True)), row.evidence_id)
        for row in stored
        if current.get(row.evidence_id) == row.revision
    ]
    return [i for _, i in sorted(scored, key=lambda p: (-p[0], p[1]))][:depth]


def rrf(rankings: dict[str, list[str]], k: int = RRF_K) -> list[tuple[str, float, dict[str, int]]]:
    """Reciprocal rank fusion: score = Σ 1 / (k + rank)."""
    fused: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, ranking in rankings.items():
        for position, evidence_id in enumerate(ranking, start=1):
            fused[evidence_id] = fused.get(evidence_id, 0.0) + 1.0 / (k + position)
            ranks.setdefault(evidence_id, {})[name] = position
    order = sorted(fused, key=lambda i: (-fused[i], i))
    return [(i, fused[i], ranks[i]) for i in order]


def search(
    session: Session,
    query: str,
    embedder: Embedder | None = None,
    k: int = 10,
    mode: Mode = Mode.HYBRID,
    depth: int = 50,
) -> list[Hit]:
    """Top-`k` citable records for `query`.

    `depth` is how far down each retriever's list fusion looks. It is wider
    than `k` so that a record ranked modestly by both retrievers can still
    outrank one ranked highly by only one.
    """
    if mode is not Mode.BM25 and embedder is None:
        raise ValueError(f"mode '{mode}' needs an embedder")
    records = citable_records(session)
    if not records:
        return []
    by_id = {r.id: r for r in records}
    ids = [r.id for r in records]

    rankings: dict[str, list[str]] = {}
    if mode in (Mode.HYBRID, Mode.BM25):
        index = BM25([query_terms(document_text(r)) for r in records])
        rankings["bm25"] = _ranked(ids, index.scores(query_terms(query)))[:depth]
    if mode in (Mode.HYBRID, Mode.DENSE):
        assert embedder is not None
        rankings["dense"] = _dense_ranking(session, records, query, embedder, depth)

    return [
        Hit(evidence_id=i, statement=by_id[i].statement, score=score, ranks=ranks)
        for i, score, ranks in rrf(rankings)[:k]
    ]
