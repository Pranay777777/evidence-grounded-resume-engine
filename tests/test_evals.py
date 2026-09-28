"""Golden set, labelling, metrics and the evals CLI — no LLM, no NLI model."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from grounded.config import get_settings
from grounded.evals import __main__ as cli
from grounded.evals.collect import MAX_CONSECUTIVE_FAILURES, PinRequiredError, collect, same_model
from grounded.evals.golden import (
    GoldenItem,
    Job,
    JobSet,
    item_id,
    merge,
    read_items,
    read_jobs,
    write_items,
)
from grounded.evals.label import label
from grounded.evals.metrics import check, evaluate, gate, markdown, tone_ok
from grounded.generation.llm import FREE_ROUTER, ChatResponse, LLMError
from grounded.retrieval.embedding import get_embedder
from grounded.verification.nli import Label, Verdict

CORPUS = Path(__file__).parent / "fixtures" / "retrieval_corpus.yaml"


def item(ident: str, jd: str = "j1", **kwargs: Any) -> GoldenItem:
    fields: dict[str, Any] = {
        "id": ident,
        "jd_id": jd,
        "model": "test/model",
        "prompt_version": "generate-v1",
        "text": "Built Airflow DAGs.",
        "evidence_ids": ["r1"],
        "premise": "Built Airflow DAGs.",
    }
    fields.update(kwargs)
    return GoldenItem(**fields)


class Scripted:
    """Entailment per hypothesis; anything unlisted is entailed at 0.99."""

    name = "scripted-nli"

    def __init__(self, verdicts: dict[str, Verdict] | None = None) -> None:
        self.verdicts = verdicts or {}
        self.calls: list[str] = []

    def check(self, premise: str, hypothesis: str) -> Verdict:
        self.calls.append(hypothesis)
        return self.verdicts.get(hypothesis, Verdict(Label.ENTAILMENT, 0.99))


# ── golden-set IO ─────────────────────────────────────────────────────────────


def test_items_round_trip_through_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "golden.jsonl"
    labelled = item(
        "a", supported=False, note="scope", labelled_by="me", labelled_at=datetime.now(UTC)
    )
    write_items(path, [labelled, item("b", text="Wrote dbt models — über-fast.")])
    assert read_items(path) == [labelled, item("b", text="Wrote dbt models — über-fast.")]
    assert "über" in path.read_text(encoding="utf-8")  # not \\u-escaped: diffs stay readable
    assert b"\r\n" not in path.read_bytes()


def test_a_missing_file_is_an_empty_set_and_blank_lines_are_ignored(tmp_path: Path) -> None:
    assert read_items(tmp_path / "none.jsonl") == []
    path = tmp_path / "golden.jsonl"
    path.write_text("\n" + item("a").model_dump_json() + "\n\n", encoding="utf-8")
    assert [i.id for i in read_items(path)] == ["a"]


def test_a_failed_write_leaves_the_old_file_and_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "golden.jsonl"
    write_items(path, [item("a")])
    before = path.read_bytes()

    def boom(*args: Any, **kwargs: Any) -> str:
        raise OSError("disk full")

    monkeypatch.setattr(json, "dumps", boom)
    with pytest.raises(OSError, match="disk full"):
        write_items(path, [item("a"), item("b")])
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["golden.jsonl"]


def test_job_sets_are_validated(tmp_path: Path) -> None:
    path = tmp_path / "jds.yaml"
    path.write_text("jds:\n  - {id: j1, title: T, text: X, relevant: [r1]}\n", encoding="utf-8")
    assert read_jobs(path).jds[0].relevant == ["r1"]
    path.write_text("jds: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least 1"):
        read_jobs(path)


def test_the_shipped_job_descriptions_only_label_real_records() -> None:
    import yaml

    corpus = yaml.safe_load(Path("benchmarks/retrieval/corpus.yaml").read_text(encoding="utf-8"))
    ids = {e["id"] for e in corpus["evidence"]}
    jobs = read_jobs(cli.JOBS)
    assert len(jobs.jds) == 20
    assert len({j.id for j in jobs.jds}) == 20
    assert all(j.relevant and set(j.relevant) <= ids for j in jobs.jds)


def test_item_ids_are_content_addressed() -> None:
    first = item_id("jd01", "m/a", "Built X.")
    assert first == item_id("jd01", "m/a", "Built X.")
    assert first.startswith("jd01-")
    assert first != item_id("jd01", "m/b", "Built X.")
    assert first != item_id("jd01", "m/a", "Built Y.")


def test_merge_never_drops_or_overwrites_existing_items() -> None:
    old = [item("a", supported=True, labelled_by="me"), item("b")]
    new = [item("a", text="changed"), item("b", text="changed"), item("c"), item("c")]
    merged, added = merge(old, new)
    assert added == 1
    assert [i.id for i in merged] == ["a", "b", "c"]
    assert merged[0].supported is True and merged[1].text == "Built Airflow DAGs."


# ── collect ───────────────────────────────────────────────────────────────────


class FakeClient:
    """Cites a retrieved record plus one ID it was never given.

    Jobs mentioning BROKEN get invalid output; RATE jobs get a provider error.
    """

    def __init__(self, model: str = "pinned/model:free", limited: bool = False) -> None:
        self.model = model
        self.limited = limited
        self.calls = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> ChatResponse:
        self.calls += 1
        prompt = messages[1]["content"]
        if "RATE" in prompt or self.limited:
            raise LLMError("provider returned 429: Provider returned error")
        if "BROKEN" in prompt:
            return ChatResponse(content="not json", tool_arguments=None, model=self.model)
        cited = "ex-delta-merge" if "ex-delta-merge" in prompt else "ex-hnsw-index"
        bullets = [
            {
                "text": "Implemented incremental loads with Delta Lake MERGE.",
                "evidence_ids": [cited],
            },
            {"text": "Ran Kubernetes clusters.", "evidence_ids": [cited, "made-up-record"]},
        ]
        return ChatResponse(
            content=None, tool_arguments=json.dumps({"bullets": bullets}), model=self.model
        )


JOBS = JobSet(
    jds=[
        Job(id="j1", title="DE", text="Delta Lake incremental loads and Kubernetes."),
        Job(id="j2", title="Broken", text="BROKEN job description about Delta Lake."),
    ]
)


def test_collect_refuses_the_random_router() -> None:
    with pytest.raises(PinRequiredError, match="pinned model"):
        collect(CORPUS, JOBS, FakeClient(FREE_ROUTER), get_embedder("hashing"))


def test_collect_keeps_every_bullet_ungated_with_its_premise() -> None:
    progress: list[str] = []
    saved: list[list[GoldenItem]] = []
    result = collect(
        CORPUS,
        JOBS,
        FakeClient(),
        get_embedder("hashing"),
        save=saved.append,
        progress=progress.append,
    )
    items = result.items
    assert [i.jd_id for i in items] == ["j1", "j1"]
    assert saved == [items]  # saved per job, as soon as it arrives
    assert (result.done, result.failed, result.stopped) == (["j1"], ["j2"], False)
    faithful, invented = items
    assert faithful.model == "pinned/model:free" and faithful.prompt_version == "generate-v1"
    assert faithful.evidence_ids == ["ex-delta-merge"] and faithful.unknown_ids == []
    assert faithful.premise.startswith("Implemented incremental loads with Delta Lake MERGE.")
    assert "This was part of the Example lakehouse project." in faithful.premise
    assert invented.unknown_ids == ["made-up-record"]  # kept: the gate must see it
    assert all(i.supported is None for i in items)
    assert progress[0] == "  j1: 2 bullet(s) from pinned/model:free"
    assert progress[1].startswith("  j2: no draft")


def rate_jobs(n: int) -> JobSet:
    return JobSet(jds=[Job(id=f"r{i}", title="R", text="RATE Delta Lake loads.") for i in range(n)])


def test_a_provider_error_skips_the_job_and_the_run_continues() -> None:
    jobs = JobSet(jds=[*rate_jobs(1).jds, *JOBS.jds])
    progress: list[str] = []
    result = collect(CORPUS, jobs, FakeClient(), get_embedder("hashing"), progress=progress.append)
    assert result.done == ["j1"] and result.failed == ["r0", "j2"]
    assert progress[0] == "  r0: provider error (provider returned 429: Provider returned error)"


def test_consecutive_provider_errors_stop_the_run_without_spending_more_calls() -> None:
    client = FakeClient(limited=True)
    result = collect(CORPUS, rate_jobs(5), client, get_embedder("hashing"))
    assert result.stopped and len(result.failed) == 5 and result.items == []
    assert client.calls == MAX_CONSECUTIVE_FAILURES  # jobs after the streak cost nothing


def test_finished_jobs_are_skipped() -> None:
    client = FakeClient()
    result = collect(CORPUS, JOBS, client, get_embedder("hashing"), skip=frozenset({"j1"}))
    assert result.skipped == ["j1"] and result.done == []


def test_model_ids_match_with_or_without_the_free_suffix() -> None:
    assert same_model("google/gemma-4-31b-it", "google/gemma-4-31b-it:free")
    assert not same_model("google/gemma-4-31b-it", "qwen/qwen3.8-27b:free")


# ── labelling ─────────────────────────────────────────────────────────────────


def answers(*replies: str) -> Any:
    queue = list(replies)
    return lambda prompt: queue.pop(0)


def test_labelling_saves_each_answer_and_resumes(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    write_items(
        path,
        [
            item("done", supported=True, labelled_by="earlier"),
            item("a"),
            item("b", unknown_ids=["ghost"]),
            item("c", premise=""),
        ],
    )
    shown: list[str] = []
    jobs = JobSet(jds=[Job(id="j1", title="Data Engineer", text="x")])
    replies = answers("?", "y", "n", "too broad", "s")
    done, left = label(path, jobs, "Pranay", replies, shown.append)
    assert (done, left) == (2, 1)
    by_id = {i.id: i for i in read_items(path)}
    assert by_id["done"].labelled_by == "earlier"
    assert by_id["a"].supported is True and by_id["a"].labelled_by == "Pranay"
    assert by_id["a"].labelled_at is not None
    assert by_id["b"].supported is False and by_id["b"].note == "too broad"
    assert by_id["c"].supported is None
    text = "\n".join(shown)
    assert "Data Engineer" in text and "3 left" in text
    assert "also cites unknown IDs: ghost" in text
    assert "(nothing resolvable was cited)" in text

    assert label(path, jobs, "Pranay", answers("q"), shown.append) == (0, 1)
    assert label(path, jobs, "Pranay", answers("Y"), shown.append) == (1, 0)


# ── metrics ───────────────────────────────────────────────────────────────────

GOLDEN_JOBS = JobSet(
    jds=[
        Job(
            id="j1",
            title="One",
            text="Orchestrate Airflow and dbt; Kubernetes a plus.",
            relevant=["r1", "r2"],
        ),
        Job(id="j2", title="Two", text="Airflow migration.", relevant=["r3"]),
        Job(id="j3", title="Unlabelled", text="Airflow.", relevant=["r1"]),
    ]
)
CORPUS_TEXT = ["Built Airflow DAGs.", "Wrote dbt models."]


def golden() -> list[GoldenItem]:
    return [
        item("a", supported=True, text="Built Airflow DAGs for the batch platform."),
        item(
            "b",
            supported=False,
            text="Wrote dbt models for finance.",
            evidence_ids=["r2"],
            premise="Wrote dbt models.",
        ),
        item(
            "c",
            supported=True,
            text="Wrote dbt models.",
            evidence_ids=["r2"],
            premise="Wrote dbt models.",
        ),
        item("d", "j2", supported=False, unknown_ids=["zz"]),
        item("e", "j2", supported=False, text="Cut cost by 40%.", premise="Cut cost."),
        item(
            "f",
            "j2",
            supported=False,
            text="Led the migration.",
            premise="Contributed to the migration.",
        ),
        item("g", "j2"),  # unlabelled: ignored
        item("h", "j2", supported=False, premise=""),
        item("i", "j3"),  # a job with no labelled bullets is skipped, not scored as zero
    ]


def test_every_metric_on_a_hand_computed_set() -> None:
    verifier = Scripted({"Wrote dbt models.": Verdict(Label.NEUTRAL, 0.10)})
    report = evaluate(golden(), GOLDEN_JOBS, verifier, 0.95, CORPUS_TEXT)
    assert (report.items, report.labelled, report.kept) == (9, 7, 2)
    assert report.raw_fabrication == pytest.approx(5 / 7)  # b d e f h
    assert report.output_fabrication == pytest.approx(1 / 2)  # b among a b
    assert report.false_accept == pytest.approx(1 / 5)
    assert report.false_reject == pytest.approx(1 / 2)  # c
    assert report.false_accepts == ["b"]
    assert report.drop_reasons == {
        "claim strength": 1,
        "neutral": 1,
        "nothing cited": 1,
        "number": 1,
        "unknown citation": 1,
    }
    # j1 cites {r1, r2} = relevant → P 1, R 1; j2 keeps nothing → no P, R 0.
    assert report.citation_precision == pytest.approx(1.0)
    assert report.citation_recall == pytest.approx(0.5)
    # j1 attainable {airflow, dbt} both used; j2 attainable {airflow} unused.
    assert report.keyword_coverage == pytest.approx(0.5)
    assert report.tone == pytest.approx(1.0)
    assert report.models == ["test/model"]
    # The model is only asked about bullets that passed every deterministic check.
    assert sorted(verifier.calls) == sorted(
        [
            "Built Airflow DAGs for the batch platform.",
            "Wrote dbt models for finance.",
            "Wrote dbt models.",
        ]
    )


def test_entailment_below_the_threshold_is_dropped() -> None:
    verifier = Scripted({"Built Airflow DAGs.": Verdict(Label.ENTAILMENT, 0.90)})
    assert gate(item("a"), verifier, 0.95).reason == "entailment 0.90"
    assert gate(item("a"), verifier, 0.85).kept


def test_an_empty_set_scores_zero_without_dividing_by_zero() -> None:
    report = evaluate([], GOLDEN_JOBS, Scripted(), 0.95, CORPUS_TEXT)
    assert report.kept == 0 and report.output_fabrication == 0.0 and report.models == []


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("Built Airflow DAGs.", True),
        ("I built Airflow DAGs.", False),
        ("Our team built DAGs.", False),
        ("built Airflow DAGs.", False),
        ("The pipeline was fast.", False),
        ("Built DAGs. Wrote tests.", False),
        ("", False),
    ],
)
def test_tone_rules(text: str, ok: bool) -> None:
    assert tone_ok(text) is ok


def test_markdown_leads_with_output_fabrication() -> None:
    report = evaluate(golden(), GOLDEN_JOBS, Scripted(), 0.95, CORPUS_TEXT)
    text = markdown(report, "scripted-nli", 0.95)
    assert text.startswith("# Evaluation — golden set\n")
    assert "7 human-labelled bullets (of 9)" in text and "`scripted-nli` ≥ 0.95" in text
    assert "| **Output fabrication rate** (unsupported among kept) | **33%** |" in text
    assert "False accepts (read these first): b" in text
    empty = markdown(evaluate([], GOLDEN_JOBS, Scripted(), 0.95, []), "x", 0.95)
    assert "Drop reasons: none" in empty and "read these first): none" in empty


def test_limits_are_checked_in_both_directions() -> None:
    report = evaluate(golden(), GOLDEN_JOBS, Scripted(), 0.95, CORPUS_TEXT)
    assert check(report, {"max": {"false_reject": 0.5}, "min": {"tone": 1.0}}) == []
    assert check(report, {"max": {"false_accept": 0.1}, "min": {"citation_recall": 0.9}}) == [
        "false_accept 0.20 > 0.1",
        "citation_recall 0.50 < 0.9",
    ]
    assert check(report, {"max": None}) == []  # type: ignore[dict-item]


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ({"maximum": {"tone": 1}}, "under 'max' or 'min'"),
        ({"max": {"fabrication": 0.1}}, "'fabrication' is not a numeric metric"),
        ({"max": {"kept": 3}}, "'kept' is not a numeric metric"),
    ],
)
def test_bad_limits_are_errors(limits: dict[str, dict[str, float]], message: str) -> None:
    report = evaluate(golden(), GOLDEN_JOBS, Scripted(), 0.95, CORPUS_TEXT)
    with pytest.raises(ValueError, match=message):
        check(report, limits)


def test_the_shipped_limits_file_is_valid() -> None:
    import yaml

    limits = yaml.safe_load(Path("benchmarks/golden/limits.yaml").read_text(encoding="utf-8"))
    report = evaluate(golden(), GOLDEN_JOBS, Scripted(), 0.95, CORPUS_TEXT)
    check(report, limits)  # raises on unknown names or bounds


# ── CLI ───────────────────────────────────────────────────────────────────────


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    jobs = tmp_path / "jds.yaml"
    jobs.write_text(
        "jds:\n"
        "  - {id: j1, title: DE, relevant: [ex-delta-merge],\n"
        "     text: Delta Lake incremental loads.}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "JOBS", jobs)
    monkeypatch.setattr(cli, "CORPUS", CORPUS)
    monkeypatch.setattr(cli, "OpenAICompatibleClient", lambda key, model, url: FakeClient(model))
    monkeypatch.setattr(cli, "get_verifier", lambda name, cache_dir=None: Scripted())
    monkeypatch.setenv("EMBEDDER", "hashing")
    get_settings.cache_clear()
    yield tmp_path / "golden.jsonl"
    get_settings.cache_clear()


def test_cli_collect_then_collect_again_adds_nothing(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["collect", "--model", "pinned/model:free", "--out", str(workspace)]) == 0
    assert "2 bullet(s) from 1 job(s) saved" in capsys.readouterr().out
    assert cli.main(["collect", "--model", "pinned/model", "--out", str(workspace)]) == 0
    out = capsys.readouterr().out
    assert "0 bullet(s) from 0 job(s)" in out and "1 job(s) already collected" in out
    assert len(read_items(workspace)) == 2
    assert cli.main(["collect", "--model", "second/model:free", "--out", str(workspace)]) == 0
    assert len(read_items(workspace)) == 4


def test_cli_collect_after_a_rate_limit_resumes(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli, "OpenAICompatibleClient", lambda key, model, url: FakeClient(model, limited=True)
    )
    assert cli.main(["collect", "--model", "pinned/model:free", "--out", str(workspace)]) == 1
    assert "1 job(s) not collected: j1" in capsys.readouterr().err
    assert not workspace.exists()
    monkeypatch.setattr(cli, "OpenAICompatibleClient", lambda key, model, url: FakeClient(model))
    assert cli.main(["collect", "--model", "pinned/model:free", "--out", str(workspace)]) == 0
    assert len(read_items(workspace)) == 2


def test_cli_collect_needs_a_pin(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["collect", "--model", FREE_ROUTER, "--out", str(workspace)]) == 1
    assert "pinned model" in capsys.readouterr().err
    assert not workspace.exists()


def test_cli_label_and_run(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["run", "--file", str(workspace)]) == 1
    assert "collect and label first" in capsys.readouterr().err

    cli.main(["collect", "--model", "pinned/model:free", "--out", str(workspace)])
    monkeypatch.setattr("builtins.input", answers("y", "n", ""))
    assert cli.main(["label", "--by", "Pranay", "--file", str(workspace)]) == 0
    assert "labelled 2 this session; 0 still unlabelled" in capsys.readouterr().out

    out = workspace.parent / "results" / "eval.md"
    assert cli.main(["run", "--file", str(workspace), "--out", str(out)]) == 0
    written = out.read_text(encoding="utf-8")
    assert "**0%**" in written  # the invented citation is dropped by the gate
    assert "unknown citation 1" in written

    limits = workspace.parent / "limits.yaml"
    limits.write_text("max: {output_fabrication: 0.0}\nmin: {tone: 1.0}\n", encoding="utf-8")
    assert cli.main(["run", "--file", str(workspace), "--check", str(limits)]) == 0
    limits.write_text("min: {citation_recall: 1.1}\n", encoding="utf-8")
    assert cli.main(["run", "--file", str(workspace), "--check", str(limits)]) == 1
    assert "REGRESSION: citation_recall" in capsys.readouterr().err
    limits.write_text("max: {nonsense: 1}\n", encoding="utf-8")
    assert cli.main(["run", "--file", str(workspace), "--check", str(limits)]) == 2
    assert "not a numeric metric" in capsys.readouterr().err
