"""The semantic cache (ADR-013) and its benchmark - no model is called."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from grounded.config import get_settings
from grounded.evidence.__main__ import main as evidence_main
from grounded.generation import __main__ as cli
from grounded.generation.cache import SemanticCache, Stats, cosine, request_key
from grounded.generation.cache_bench import (
    CachePairSet,
    Scored,
    markdown,
    read_pairs,
    score,
    sweep,
)
from grounded.generation.llm import ChatResponse
from grounded.generation.schema import Draft
from grounded.retrieval.__main__ import main as retrieval_main
from grounded.retrieval.embedding import get_embedder
from grounded.verification.nli import Label, Verdict

FIXTURE = Path(__file__).parent / "fixtures" / "retrieval_corpus.yaml"
DRAFT = Draft.model_validate(
    {"bullets": [{"text": "Built Airflow DAGs.", "evidence_ids": ["ex-delta-merge"]}]}
)


class TableEmbedder:
    """Vectors by first word, so similarity is chosen by the test."""

    name = "table"
    dim = 2

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self.table[t.split()[0]] for t in texts]


EMBED = TableEmbedder({"alpha": [1.0, 0.0], "close": [0.99, 0.141], "far": [0.0, 1.0]})


def test_request_keys_ignore_evidence_order_but_nothing_else() -> None:
    key = request_key("m", "fp", [("a", 1), ("b", 2)])
    assert key == request_key("m", "fp", [("b", 2), ("a", 1)])
    assert key != request_key("m", "fp", [("a", 1), ("b", 3)])  # an edited record
    assert key != request_key("other", "fp", [("a", 1), ("b", 2)])
    assert key != request_key("m", "fp2", [("a", 1), ("b", 2)])


def test_cosine() -> None:
    assert cosine([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 3.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_exact_and_similar_requests_hit_and_the_rest_miss(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    cache = SemanticCache(path, EMBED, threshold=0.95)
    assert cache.lookup("k", "alpha job") is None  # empty cache: no embedding call
    assert EMBED.calls == 0
    cache.store("k", "alpha   JOB", DRAFT, "m", "generate-v1", tokens=120)

    exact = cache.lookup("k", "Alpha job")
    assert exact is not None and exact.similarity == 1.0 and exact.entry.draft == DRAFT
    similar = cache.lookup("k", "close job")
    assert similar is not None and similar.similarity == pytest.approx(0.99, abs=0.01)
    assert cache.lookup("k", "far job") is None
    assert cache.lookup("other-key", "alpha job") is None

    reopened = SemanticCache(path, EMBED, threshold=0.95)
    assert reopened.stats() == Stats(entries=1, hits=2, tokens_saved=240)
    assert reopened.stats().hit_rate == pytest.approx(2 / 3)
    assert SemanticCache(tmp_path / "none.jsonl", EMBED, 0.95).stats().hit_rate == 0.0
    assert [p.name for p in tmp_path.iterdir()] == ["cache.jsonl"]


def test_a_failed_write_leaves_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = SemanticCache(tmp_path / "cache.jsonl", EMBED, 0.95)

    def boom(*args: Any, **kwargs: Any) -> str:
        raise OSError("disk full")

    monkeypatch.setattr(json, "dumps", boom)
    with pytest.raises(OSError):
        cache.store("k", "alpha", DRAFT, "m", "generate-v1", tokens=1)
    assert list(tmp_path.iterdir()) == []


# --- benchmark -------------------------------------------------------------------


def test_the_shipped_pairs_are_balanced() -> None:
    pairs = read_pairs(Path("benchmarks/cache/pairs.yaml"))
    assert len(pairs.pairs) == 24 and sum(p.same for p in pairs.pairs) == 12
    assert len({p.id for p in pairs.pairs}) == 24


PAIRS = CachePairSet.model_validate(
    {
        "pairs": [
            {
                "id": "s1",
                "kind": "format",
                "same": True,
                "a": "Delta Lake MERGE",
                "b": "delta  lake merge",
            },
            {
                "id": "s2",
                "kind": "paraphrase",
                "same": True,
                "a": "Delta Lake MERGE",
                "b": "HNSW index",
            },
            {
                "id": "d1",
                "kind": "different",
                "same": False,
                "a": "Delta Lake MERGE",
                "b": "HNSW index",
            },
        ]
    }
)


def test_score_and_sweep() -> None:
    scored = score(PAIRS, FIXTURE, get_embedder("hashing"), k=3)
    assert scored[0].similarity == 1.0 and scored[0].same_evidence
    assert scored[1].similarity < 0.8
    rows = {r.threshold: r for r in sweep(scored)}
    assert rows[0.95].hit_rate == 0.5 and rows[0.95].false_hits == 0.0
    assert rows[0.95].hit_rate_with_evidence == 0.5


def test_sweep_separates_similarity_from_the_evidence_rule() -> None:
    p = PAIRS.pairs
    scored = [Scored(p[0], 0.99, False), Scored(p[1], 0.99, True), Scored(p[2], 0.96, False)]
    rows = {r.threshold: r for r in sweep(scored)}
    assert rows[0.95].hit_rate == 1.0 and rows[0.95].false_hits == 1.0
    assert rows[0.95].hit_rate_with_evidence == 0.5 and rows[0.95].false_hits_with_evidence == 0.0


def test_the_report_projects_savings_only_from_recorded_tokens() -> None:
    scored = [Scored(PAIRS.pairs[0], 1.0, True), Scored(PAIRS.pairs[2], 0.1, False)]
    text = markdown(scored, "hashing", 0.95, None, 0.0)
    assert "| 0.95 (configured) | 100% | 0% | 100% | 0% |" in text
    assert "no token counts recorded yet" in text
    priced = markdown(scored, "hashing", 0.95, 1500.0, 2.0)
    assert "saving about 150,000 tokens per 100 requests" in priced and "$0.3000" in priced
    free = markdown(scored, "hashing", 0.95, 1500.0, 0.0)
    assert "(free models cost $0)" in free
    assert "not in the sweep" in markdown(scored, "hashing", 0.93, None, 0.0)


# --- CLI -------------------------------------------------------------------------


class CountingClient:
    calls = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.model = "fake/model"

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        CountingClient.calls += 1
        args = {
            "bullets": [
                {
                    "text": "Implemented incremental loads with Delta Lake MERGE.",
                    "evidence_ids": ["ex-delta-merge"],
                }
            ]
        }
        return ChatResponse(
            content=None,
            tool_arguments=json.dumps(args),
            model="fake/model",
            usage={"prompt_tokens": 900, "completion_tokens": 100},
        )


class Entailed:
    name = "fake-nli"

    def check(self, premise: str, hypothesis: str) -> Verdict:
        return Verdict(Label.ENTAILMENT, 0.99)


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'store.db'}")
    monkeypatch.setenv("EMBEDDER", "hashing")
    monkeypatch.setenv("SEMANTIC_CACHE_PATH", str(tmp_path / "cache.jsonl"))
    monkeypatch.setattr(cli, "get_verifier", lambda name, cache_dir=None: Entailed())
    monkeypatch.setattr(cli, "make_client", CountingClient)
    CountingClient.calls = 0
    get_settings.cache_clear()
    evidence_main(["load", str(FIXTURE)])
    retrieval_main(["index"])
    jd = tmp_path / "jd.txt"
    jd.write_text("Data engineer with Delta Lake incremental loads.", encoding="utf-8")
    yield jd
    get_settings.cache_clear()


def test_draft_with_cache_reuses_a_repeat_and_still_gates_it(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    assert cli.main(["draft", "--jd", str(store), "--cache"]) == 0
    first = capsys.readouterr().out
    assert CountingClient.calls == 1 and "cache hit" not in first
    store.write_text("DATA ENGINEER with Delta Lake   incremental loads.", encoding="utf-8")
    assert cli.main(["draft", "--jd", str(store), "--cache"]) == 0
    second = capsys.readouterr().out
    assert CountingClient.calls == 1
    assert "| cache hit (similarity 1.000)" in second and "1 kept, 0 dropped." in second
    assert cli.main(["draft", "--jd", str(store)]) == 0  # no --cache: always calls
    assert CountingClient.calls == 2
    capsys.readouterr()
    assert cli.main(["cache-stats"]) == 0
    assert (
        "1 cached draft(s), 1 hit(s), hit rate 50%, 1,000 tokens saved" in capsys.readouterr().out
    )
    assert cli.main(["draft", "--jd", str(store), "--prompt", "v9"]) == 1
    assert "unknown prompt version" in capsys.readouterr().err


def test_prompts_lists_the_registry(capsys: pytest.CaptureFixture[str]) -> None:
    get_settings.cache_clear()
    assert cli.main(["prompts"]) == 0
    out = capsys.readouterr().out
    assert "generate-v1" in out and "7a4e4c0acb2b  (default)" in out


def test_cache_bench_writes_its_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("EMBEDDER", "hashing")
    get_settings.cache_clear()
    pairs = tmp_path / "pairs.yaml"
    pairs.write_text(json.dumps(PAIRS.model_dump()), encoding="utf-8")
    runs = tmp_path / "runs.jsonl"
    runs.write_text(
        json.dumps(
            {
                "jd_id": "j",
                "model": "m",
                "requested": "m",
                "prompt_version": "generate-v1",
                "prompt_fingerprint": "fp",
                "attempts": 1,
                "prompt_tokens": 800,
                "completion_tokens": 200,
                "latency_s": 2.0,
                "bullets": 3,
                "at": "2026-09-29T00:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "semantic-cache.md"
    args = ["cache-bench", str(pairs), "--corpus", str(FIXTURE), "--runs", str(runs), "-k", "3"]
    assert cli.main([*args, "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "3 labelled job-description pairs (2 the same request, 1 different)" in text
    assert "mean 1,000 tokens per draft" in text
    capsys.readouterr()
    get_settings.cache_clear()
