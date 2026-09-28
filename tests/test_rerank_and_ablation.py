"""Reranking and the retrieval ablation harness."""

from __future__ import annotations

import builtins
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from grounded.config import get_settings
from grounded.evidence.models import Base
from grounded.evidence.store import load, read_file
from grounded.retrieval.__main__ import main
from grounded.retrieval.ablation import (
    CONFIGS,
    Benchmark,
    Config,
    evaluate,
    markdown,
    read_benchmark,
    run,
    select,
)
from grounded.retrieval.embedding import HashingEmbedder
from grounded.retrieval.index import embed_pending
from grounded.retrieval.rerank import CrossEncoderReranker, get_reranker
from grounded.retrieval.search import Mode, search

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path(__file__).parent / "fixtures" / "retrieval_corpus.yaml"
BENCH = ROOT / "benchmarks" / "retrieval"


class Prefers:
    """A stand-in cross-encoder that scores documents containing a word highest."""

    name = "fake"

    def __init__(self, word: str) -> None:
        self.word = word

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        return [1.0 if self.word in d.lower() else 0.0 for d in documents]


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        load(s, read_file(CORPUS))
        embed_pending(s, HashingEmbedder())
        yield s


# --- reranking -------------------------------------------------------------


def test_the_reranker_decides_the_final_order(session: Session) -> None:
    query = "Delta Lake incremental loads"
    plain = search(session, query, HashingEmbedder(), k=4)
    reranked = search(session, query, HashingEmbedder(), k=4, reranker=Prefers("power bi"))
    assert plain[0].evidence_id == "ex-delta-merge"
    assert reranked[0].evidence_id == "ex-powerbi-report"
    assert reranked[0].ranks["rerank"] == 1
    assert "bm25" in reranked[1].ranks or "dense" in reranked[1].ranks


def test_reranking_only_reorders_retrieved_candidates(session: Session) -> None:
    hits = search(session, "anything", HashingEmbedder(), k=10, reranker=Prefers("kubernetes"))
    assert "ex-unverified-claim" not in [h.evidence_id for h in hits]


def test_the_cross_encoder_adapter_returns_floats() -> None:
    class FakeModel:
        def rerank(self, query: str, documents: list[str]) -> Iterator[float]:
            yield from (0.25 for _ in documents)

    assert CrossEncoderReranker(model=FakeModel()).score("q", ["a", "b"]) == [0.25, 0.25]


def test_rerankers_without_the_extra_say_how_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    real = builtins.__import__

    def no_fastembed(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("fastembed"):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_fastembed)
    with pytest.raises(ImportError, match=r"\[embeddings\]"):
        get_reranker("minilm")


def test_unknown_rerankers_are_refused() -> None:
    with pytest.raises(ValueError, match="choose one of"):
        get_reranker("colbert")


# --- ablation --------------------------------------------------------------


def test_the_shipped_benchmark_is_well_formed() -> None:
    benchmark = read_benchmark(BENCH / "queries.yaml")
    corpus = {e.id for e in read_file(BENCH / "corpus.yaml").evidence}
    assert len(benchmark.queries) == 20
    for q in benchmark.queries:
        assert set(q.relevant) <= corpus, f"{q.id} labels a record that does not exist"


def test_recall_and_mrr_are_computed_correctly() -> None:
    benchmark = Benchmark.model_validate(
        {
            "queries": [
                {"id": "a", "query": "x", "relevant": ["r1", "r2"]},
                {"id": "b", "query": "y", "relevant": ["r3"]},
            ]
        }
    )
    ranked = [["r1", "z", "r2"], ["z", "z2", "z3"]]
    result = evaluate(ranked, benchmark, Config("t", Mode.BM25))
    assert result.recall[1] == pytest.approx((0.5 + 0) / 2)
    assert result.recall[3] == pytest.approx((1.0 + 0) / 2)
    assert result.mrr == pytest.approx((1.0 + 0) / 2)
    assert result.misses == {"b": ["r3"]}


def test_the_ablation_runs_and_reports_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(name: str) -> Any:
        if name == "hashing":
            return HashingEmbedder()
        raise ImportError("not here")

    monkeypatch.setattr("grounded.retrieval.ablation.get_embedder", unavailable)
    benchmark = read_benchmark(BENCH / "queries.yaml")
    configs = select(["BM25", "Hybrid / hashing", "Dense / bge-small"])
    results, skipped = run(BENCH / "corpus.yaml", benchmark, configs)
    assert [r.config.name for r in results] == ["BM25", "Hybrid / hashing"]
    assert skipped == [("Dense / bge-small", "not here")]
    assert all(0 <= r.recall[10] <= 1 for r in results)
    report = markdown(results, skipped, benchmark)
    assert "| BM25 |" in report and "Skipped (model unavailable): Dense / bge-small" in report
    assert "Synthetic benchmark" in report


def test_select_by_name_and_refuse_unknown() -> None:
    assert select(None) == CONFIGS
    assert [c.name for c in select(["bm25"])] == ["BM25"]
    with pytest.raises(ValueError, match="unknown configuration"):
        select(["telepathy"])


def test_the_ablate_command_writes_a_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'store.db'}")
    get_settings.cache_clear()
    out = tmp_path / "results" / "ablation.md"
    try:
        code = main(
            [
                "ablate",
                str(BENCH / "queries.yaml"),
                "--corpus",
                str(BENCH / "corpus.yaml"),
                "--out",
                str(out),
                "--configs",
                "BM25",
            ]
        )
    finally:
        get_settings.cache_clear()
    assert code == 0
    assert "| BM25 |" in out.read_text(encoding="utf-8")
    assert "Recall@10" in capsys.readouterr().out
