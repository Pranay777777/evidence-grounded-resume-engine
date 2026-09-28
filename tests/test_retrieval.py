"""Embedding, indexing and hybrid retrieval."""

from __future__ import annotations

import builtins
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session

from grounded.evidence.enums import VerificationMethod
from grounded.evidence.models import Base, Evidence, EvidenceEmbedding
from grounded.evidence.schema import EvidenceFile
from grounded.evidence.store import load, read_file, verify
from grounded.retrieval.chunking import document_text
from grounded.retrieval.embedding import (
    DIM,
    FastEmbedEmbedder,
    HashingEmbedder,
    get_embedder,
    tokens,
)
from grounded.retrieval.index import embed_pending
from grounded.retrieval.search import (
    BM25,
    MAX_QUERY_CHARS,
    Mode,
    query_terms,
    rrf,
    search,
)

CORPUS = Path(__file__).parent / "fixtures" / "retrieval_corpus.yaml"
EMBEDDER = HashingEmbedder()


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def _fk(conn: object, _: object) -> None:
        cur = conn.cursor()  # type: ignore[attr-defined]
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(engine)
    with Session(engine) as s:
        load(s, read_file(CORPUS))
        embed_pending(s, EMBEDDER)
        yield s


def ids(hits: list[Any]) -> list[str]:
    return [h.evidence_id for h in hits]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


# --- tokens and embedders -------------------------------------------------


def test_tokens_keep_technical_terms_whole() -> None:
    assert tokens("C++, C#, Node.js and Python 3.12!") == [
        "c++",
        "c#",
        "node.js",
        "and",
        "python",
        "3.12",
    ]


def test_hashing_is_deterministic_normalised_and_the_right_size() -> None:
    a, b = EMBEDDER.embed(["delta lake merge", "delta lake merge"])
    assert a == b
    assert len(a) == DIM
    assert math.isclose(math.sqrt(sum(v * v for v in a)), 1.0)


def test_hashing_puts_shared_vocabulary_closer() -> None:
    base, near, far = EMBEDDER.embed(
        ["incremental delta lake loads", "delta lake incremental merge", "finance power bi report"]
    )
    assert cosine(base, near) > cosine(base, far)


def test_empty_text_embeds_to_zero_not_nan() -> None:
    assert EMBEDDER.embed([""])[0] == [0.0] * DIM


def test_unknown_embedders_are_refused() -> None:
    with pytest.raises(ValueError, match="choose one of"):
        get_embedder("word2vec")


def test_hashing_is_available_by_name() -> None:
    assert get_embedder("hashing").name == "hashing"


def test_bge_without_the_extra_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    real = builtins.__import__

    def no_fastembed(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("fastembed"):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_fastembed)
    with pytest.raises(ImportError, match=r"\[embeddings\]"):
        get_embedder("bge-small")


def test_bge_adapter_returns_plain_float_lists() -> None:
    class FakeModel:
        def embed(self, texts: list[str]) -> Iterator[Any]:
            for _ in texts:
                yield (0.5, 0.25)

    assert FastEmbedEmbedder(model=FakeModel()).embed(["a", "b"]) == [[0.5, 0.25], [0.5, 0.25]]


# --- chunking -------------------------------------------------------------


def test_document_text_adds_context_but_no_new_facts(session: Session) -> None:
    record = session.get(Evidence, "ex-delta-merge")
    assert record is not None
    text = document_text(record)
    assert text.splitlines()[0] == record.statement
    assert "Project: Example lakehouse — Delta Lake medallion platform" in text
    assert "Role: Data Engineer at Example Corp" in text  # reached through the project
    assert "Skills: delta-lake, python" in text


# --- indexing -------------------------------------------------------------


def test_only_citable_records_are_embedded(session: Session) -> None:
    embedded = set(session.scalars(select(EvidenceEmbedding.evidence_id)))
    assert "ex-unverified-claim" not in embedded
    assert len(embedded) == 4


def test_indexing_again_embeds_nothing(session: Session) -> None:
    report = embed_pending(session, EMBEDDER)
    assert report.embedded == [] and report.current == 4
    assert report.summary("hashing") == "hashing: embedded 0, already current 4"


def test_a_revised_record_is_re_embedded(session: Session) -> None:
    raw = yaml.safe_load(CORPUS.read_text())
    raw["evidence"][1]["statement"] = "Implemented incremental loads with Delta Lake MERGE and CDC."
    load(session, EvidenceFile.model_validate(raw))  # verification block re-verifies it
    report = embed_pending(session, EMBEDDER)
    assert report.embedded == ["ex-delta-merge"]
    row = session.get(EvidenceEmbedding, ("ex-delta-merge", "hashing"))
    assert row is not None and row.revision == 2


# --- BM25 and fusion -------------------------------------------------------


def test_bm25_prefers_rarer_terms() -> None:
    index = BM25([["delta", "lake"], ["delta", "pytest"], ["delta", "power"]])
    scores = index.scores(["pytest"])
    assert scores[1] > 0 and scores[0] == scores[2] == 0


def test_bm25_normalises_for_length() -> None:
    index = BM25([["merge"], ["merge", "x", "y", "z", "w", "v"]])
    short, long = index.scores(["merge"])
    assert short > long


def test_bm25_on_an_empty_corpus() -> None:
    assert BM25([]).scores(["anything"]) == []


def test_rrf_rewards_agreement() -> None:
    fused = rrf({"bm25": ["a", "b", "c"], "dense": ["b", "a", "d"]}, k=60)
    order = [i for i, _, _ in fused]
    assert set(order[:2]) == {"a", "b"}  # ranked by both
    score_a = next(s for i, s, _ in fused if i == "a")
    assert math.isclose(score_a, 1 / 61 + 1 / 62)
    assert {i: r for i, _, r in fused}["d"] == {"dense": 3}


def test_stopwords_are_not_query_terms() -> None:
    assert query_terms("The role is for a Data Engineer with Delta Lake") == [
        "role", "data", "engineer", "delta", "lake"
    ]  # fmt: skip


# --- search ----------------------------------------------------------------


def test_bm25_finds_an_exact_rare_term(session: Session) -> None:
    hits = search(session, "experience with HNSW indexes", mode=Mode.BM25, k=3)
    assert ids(hits)[0] == "ex-hnsw-index"
    assert hits[0].ranks == {"bm25": 1}


def test_dense_search_ranks_by_similarity(session: Session) -> None:
    hits = search(session, "Power BI finance reporting", EMBEDDER, mode=Mode.DENSE, k=2)
    assert ids(hits)[0] == "ex-powerbi-report"
    assert "dense" in hits[0].ranks


def test_hybrid_records_where_each_retriever_placed_a_hit(session: Session) -> None:
    hits = search(session, "Delta Lake incremental MERGE loads in Python", EMBEDDER, k=4)
    top = hits[0]
    assert top.evidence_id == "ex-delta-merge"
    assert set(top.ranks) == {"bm25", "dense"}
    assert all(h.score > 0 for h in hits)


def test_unverified_records_are_never_returned(session: Session) -> None:
    for mode in Mode:
        hits = search(session, "Kubernetes migration clusters", EMBEDDER, mode=mode, k=10)
        assert "ex-unverified-claim" not in ids(hits)


def test_a_record_edited_since_embedding_is_not_dense_retrievable(session: Session) -> None:
    raw = yaml.safe_load(CORPUS.read_text())
    raw["evidence"][3]["statement"] = "Built Power BI reports and paginated reports for finance."
    load(session, EvidenceFile.model_validate(raw))  # rev 2, re-verified, not re-embedded
    hits = search(session, "Power BI finance reporting", EMBEDDER, mode=Mode.DENSE, k=10)
    assert "ex-powerbi-report" not in ids(hits)


def test_an_edited_and_unverified_record_disappears(session: Session) -> None:
    raw = yaml.safe_load(CORPUS.read_text())
    raw["evidence"][0]["statement"] = "Added an HNSW index and tuned ef_search for recall."
    del raw["evidence"][0]["verification"]
    load(session, EvidenceFile.model_validate(raw))
    assert "ex-hnsw-index" not in ids(search(session, "HNSW", EMBEDDER, k=10))
    verify(session, "ex-hnsw-index", VerificationMethod.SELF_ATTESTED, "tester")
    assert "ex-hnsw-index" in ids(search(session, "HNSW", mode=Mode.BM25, k=10))


def test_dense_modes_need_an_embedder(session: Session) -> None:
    with pytest.raises(ValueError, match="needs an embedder"):
        search(session, "anything", mode=Mode.HYBRID)


def test_an_empty_store_returns_nothing() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        assert search(s, "anything", EMBEDDER) == []


def test_huge_queries_are_capped(session: Session) -> None:
    """Terms past the cap are never seen. Observable in BM25, which drops
    zero-score records — dense search ranks every record regardless."""
    assert "ex-hnsw-index" in ids(search(session, "HNSW", mode=Mode.BM25, k=3))  # control
    padding = "filler " * (MAX_QUERY_CHARS // 7 + 10)
    assert search(session, padding + "HNSW", mode=Mode.BM25, k=3) == []
    assert "hnsw" not in query_terms(padding + "HNSW")
