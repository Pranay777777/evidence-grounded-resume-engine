"""Hard grounding (step 48) and the entailment verifier (step 49)."""

from __future__ import annotations

import builtins
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from grounded.config import get_settings
from grounded.evidence.models import Base, Evidence
from grounded.evidence.schema import EvidenceFile
from grounded.evidence.store import load, read_file
from grounded.generation.schema import Draft
from grounded.verification.calibration import (
    THRESHOLDS,
    PairSet,
    accepted,
    markdown,
    read_pairs,
    sweep,
)
from grounded.verification.grounding import ground
from grounded.verification.nli import (
    Label,
    NLIVerifier,
    Verdict,
    _softmax,
    get_verifier,
    label_order,
)
from grounded.verification.numbers import numbers, unsupported

ROOT = Path(__file__).resolve().parents[1]
LAKEHOUSE = ROOT / "evidence" / "drafts" / "lakehouse.yaml"
PAIRS = ROOT / "benchmarks" / "verifier" / "pairs.yaml"


class Scripted:
    """A verifier that answers from a table keyed by hypothesis."""

    name = "scripted"

    def __init__(
        self,
        answers: dict[str, Verdict] | None = None,
        default: Verdict = Verdict(Label.ENTAILMENT, 0.95),
    ) -> None:
        self.answers = answers or {}
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def check(self, premise: str, hypothesis: str) -> Verdict:
        self.calls.append((premise, hypothesis))
        return self.answers.get(hypothesis, self.default)


def draft(*bullets: tuple[str, list[str]]) -> Draft:
    return Draft.model_validate(
        {"bullets": [{"text": t, "evidence_ids": ids} for t, ids in bullets]}
    )


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    raw = yaml.safe_load(LAKEHOUSE.read_text(encoding="utf-8"))
    for record in raw["evidence"]:
        record["verification"] = {"method": "artifact", "by": "test"}
    with Session(engine) as s:
        load(s, EvidenceFile.model_validate(raw))
        yield s


def candidates(session: Session) -> dict[str, int]:
    return {r.id: r.revision for r in session.query(Evidence)}


# --- numbers ---------------------------------------------------------------------


def test_numbers_are_normalised() -> None:
    found = numbers("1.6 million rows, five tables, 1,600,000 again, Python 3.11/3.12, 2k users")
    assert {
        Decimal("1.6E+6"),
        Decimal(5),
        Decimal("3.11"),
        Decimal("3.12"),
        Decimal("2E+3"),
    } <= found


def test_an_inflated_number_is_caught() -> None:
    assert unsupported("Cut spend by 40 percent", ["Cut spend by 35 percent."], [Decimal(35)]) == {
        Decimal(40)
    }


def test_the_same_number_written_differently_is_supported() -> None:
    assert (
        unsupported(
            "processed 1.6M rows across 5 tables",
            ["The pipeline processes 1.6 million rows across five tables."],
            [],
        )
        == set()
    )


def test_a_metric_value_supports_its_number() -> None:
    assert (
        unsupported("reached 96% coverage", ["The suite covers most of the code."], [Decimal(96)])
        == set()
    )


# --- hard grounding ----------------------------------------------------------------


def test_a_faithful_bullet_is_kept_with_its_citation(session: Session) -> None:
    report = ground(
        session,
        draft(
            (
                "Set up continuous integration for the lakehouse on Ubuntu "
                "and Windows with a Postgres integration job.",
                ["lakehouse-ci-matrix"],
            )
        ),
        candidates(session),
        Scripted(),
    )
    assert len(report.kept) == 1 and not report.dropped
    kept = report.kept[0]
    assert kept.citations[0].evidence_id == "lakehouse-ci-matrix"
    assert kept.citations[0].revision == 1 and kept.citations[0].verification_method == "artifact"
    assert kept.entailment == pytest.approx(0.95)


def test_an_id_the_model_was_not_given_is_dropped(session: Session) -> None:
    report = ground(
        session,
        draft(("Led the lakehouse platform team for two years.", ["lakehouse-leadership"])),
        candidates(session),
        Scripted(),
    )
    assert "not in the evidence provided" in report.dropped[0].reason


def test_an_unverified_record_cannot_support_a_bullet(session: Session) -> None:
    record = session.get(Evidence, "lakehouse-ci-matrix")
    assert record is not None
    given = candidates(session)
    record.verification_status, record.verification_method = "unverified", None
    session.commit()
    report = ground(
        session,
        draft(("Set up CI for the lakehouse on Ubuntu and Windows.", ["lakehouse-ci-matrix"])),
        given,
        Scripted(),
    )
    assert "no longer verified" in report.dropped[0].reason


def test_a_record_changed_since_retrieval_is_refused(session: Session) -> None:
    given = candidates(session)
    record = session.get(Evidence, "lakehouse-ci-matrix")
    assert record is not None
    record.revision = 2
    session.commit()
    report = ground(
        session,
        draft(("Set up CI for the lakehouse on Ubuntu and Windows.", ["lakehouse-ci-matrix"])),
        given,
        Scripted(),
    )
    assert "changed since retrieval (rev 1 → 2)" in report.dropped[0].reason


def test_an_invented_number_is_dropped_before_the_model_is_asked(session: Session) -> None:
    verifier = Scripted()
    report = ground(
        session,
        draft(
            (
                "The lakehouse pipeline processes 3 million rows in 4 seconds.",
                ["lakehouse-pipeline-runtime"],
            )
        ),
        candidates(session),
        verifier,
    )
    # "3 million" is caught as 3,000,000 even though a bare 3 appears in the
    # evidence (as "3 GB") — which is why magnitudes are normalised.
    assert report.dropped[0].reason == "states 3000000, 4, not in the cited evidence"
    assert verifier.calls == []  # deterministic checks run first


def test_a_year_from_the_records_month_is_supported(session: Session) -> None:
    report = ground(
        session,
        draft(
            ("Set up CI for the lakehouse in 2026 on Ubuntu and Windows.", ["lakehouse-ci-matrix"])
        ),
        candidates(session),
        Scripted(),
    )
    assert report.kept


def test_neutral_means_dropped_not_kept(session: Session) -> None:
    text = (
        "Set up CI/CD for the lakehouse with GitHub Actions supporting Python 3.11/3.12 on "
        "Ubuntu and Windows, including a PostgreSQL integration job."
    )
    verifier = Scripted({text: Verdict(Label.NEUTRAL, 0.41)})
    report = ground(session, draft((text, ["lakehouse-ci-matrix"])), candidates(session), verifier)
    assert report.dropped[0].reason == (
        "not entailed by the cited evidence (neutral, entailment 0.41 < 0.95)"
    )


def test_entailment_below_the_threshold_is_dropped(session: Session) -> None:
    text = "Set up CI for the lakehouse on Ubuntu and Windows."
    report = ground(
        session,
        draft((text, ["lakehouse-ci-matrix"])),
        candidates(session),
        Scripted({text: Verdict(Label.ENTAILMENT, 0.7)}),
        threshold=0.8,
    )
    assert "0.70 < 0.8" in report.dropped[0].reason


def test_contradiction_is_named(session: Session) -> None:
    text = "The lakehouse has no CI at all."
    report = ground(
        session,
        draft((text, ["lakehouse-ci-matrix"])),
        candidates(session),
        Scripted({text: Verdict(Label.CONTRADICTION, 0.01)}),
    )
    assert report.dropped[0].reason.startswith("contradicted by the cited evidence")


def test_the_premise_is_statements_plus_project_names_only(session: Session) -> None:
    """The project link is part of the verified fact; summaries and skills are not."""
    verifier = Scripted()
    ground(
        session,
        draft(
            (
                "Built configuration-driven PII classification and keyed-HMAC masking into the "
                "lakehouse's Silver layer for the metadata-driven-lakehouse project.",
                ["lakehouse-pii-masking"],
            )
        ),
        candidates(session),
        verifier,
    )
    premise = verifier.calls[0][0]
    record = session.get(Evidence, "lakehouse-pii-masking")
    assert record is not None
    assert premise == f"{record.statement} This was part of the metadata-driven-lakehouse project."
    assert "Delta Lake" not in premise and "data-privacy" not in premise


def test_multiple_citations_form_one_premise(session: Session) -> None:
    verifier = Scripted()
    ground(
        session,
        draft(
            (
                "Released v1.0.0 of the lakehouse with 499 tests.",
                ["lakehouse-release-v1", "lakehouse-test-count"],
            )
        ),
        candidates(session),
        verifier,
    )
    assert "499 tests" in verifier.calls[0][0] and "Released v1.0.0" in verifier.calls[0][0]


def test_without_a_verifier_the_report_says_unverified(session: Session) -> None:
    report = ground(
        session,
        draft(("Set up CI for the lakehouse on Ubuntu and Windows.", ["lakehouse-ci-matrix"])),
        candidates(session),
        None,
    )
    assert not report.verified and report.kept[0].entailment is None


def test_nothing_is_rewritten(session: Session) -> None:
    """A dropped bullet leaves the report exactly as the model wrote it."""
    text = "Set up CI/CD for the lakehouse on Ubuntu and Windows."
    original = draft((text, ["lakehouse-ci-matrix"]))
    report = ground(
        session, original, candidates(session), Scripted({text: Verdict(Label.NEUTRAL, 0.2)})
    )
    assert report.dropped[0].bullet == original.bullets[0]


# --- the NLI adapter -------------------------------------------------------------------


def test_labels_come_from_the_model_config() -> None:
    config = {"id2label": {"0": "CONTRADICTION", "1": "ENTAILMENT", "2": "NEUTRAL"}}
    assert label_order(config) == [Label.CONTRADICTION, Label.ENTAILMENT, Label.NEUTRAL]


@pytest.mark.parametrize(
    "bad",
    [
        {"id2label": {"0": "yes", "1": "no", "2": "maybe"}},
        {"id2label": {"0": "entailment", "1": "neutral"}},
    ],
)
def test_unexpected_label_sets_are_refused(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        label_order(bad)


def test_softmax_is_a_distribution() -> None:
    probabilities = _softmax([2.0, 1.0, 0.1])
    assert sum(probabilities) == pytest.approx(1.0) and probabilities[0] > probabilities[1]


def test_the_adapter_feeds_the_model_and_maps_labels() -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.feed: dict[str, Any] = {}

        def get_inputs(self) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(name=n) for n in ("input_ids", "attention_mask", "token_type_ids")
            ]

        def run(self, outputs: Any, feed: dict[str, Any]) -> list[list[list[float]]]:
            self.feed = feed
            return [[[0.0, 4.0, 0.0]]]

    class Tokenizer:
        def encode(self, premise: str, hypothesis: str) -> SimpleNamespace:
            return SimpleNamespace(ids=[1, 2, 3], attention_mask=[1, 1, 1], type_ids=[0, 0, 1])

    session = FakeSession()
    verifier = NLIVerifier(
        session=session,
        tokenizer=Tokenizer(),
        labels=[Label.CONTRADICTION, Label.ENTAILMENT, Label.NEUTRAL],
        to_array=lambda values: [values],
    )
    verdict = verifier.check("premise", "hypothesis")
    assert verdict.label is Label.ENTAILMENT and verdict.entailment > 0.9
    assert set(session.feed) == {"input_ids", "attention_mask", "token_type_ids"}


def test_the_verifier_without_the_extra_says_how_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith(("onnxruntime", "huggingface_hub", "tokenizers")):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"\[embeddings\]"):
        get_verifier("nli-deberta")


def test_unknown_verifiers_are_refused() -> None:
    with pytest.raises(ValueError, match="choose one of"):
        get_verifier("gpt-judge")


# --- calibration -------------------------------------------------------------------------


def test_the_shipped_pairs_are_well_formed() -> None:
    pairs = read_pairs(PAIRS)
    assert len(pairs.pairs) == 21
    assert {p.id for p in pairs.pairs if p.source == "real-draft"} == {
        "real-cicd", "real-developed", "real-delta"
    }  # fmt: skip
    assert not any(p.supported for p in pairs.pairs if p.source == "real-draft")


def test_the_sweep_counts_both_error_types() -> None:
    pairs = PairSet.model_validate(
        {
            "pairs": [
                {"id": "a", "source": "s", "premise": "p", "hypothesis": "h", "supported": True},
                {"id": "b", "source": "s", "premise": "p", "hypothesis": "h", "supported": False},
            ]
        }
    )
    verdicts = [Verdict(Label.ENTAILMENT, 0.75), Verdict(Label.ENTAILMENT, 0.65)]
    rows = {r.threshold: r for r in sweep(pairs, verdicts)}
    assert rows[0.6].false_accept == 1.0 and rows[0.6].false_reject == 0.0
    assert rows[0.7].false_accept == 0.0 and rows[0.7].false_reject == 0.0
    assert rows[0.8].false_reject == 1.0 and rows[0.8].accuracy == 0.5
    assert [r.threshold for r in sweep(pairs, verdicts)] == list(THRESHOLDS)


def test_neutral_is_never_accepted_whatever_the_threshold() -> None:
    assert not accepted(Verdict(Label.NEUTRAL, 0.99), 0.5)


def test_the_calibration_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from grounded.verification import __main__ as cli

    monkeypatch.setattr(cli, "get_verifier", lambda name, cache_dir=None: Scripted())
    get_settings.cache_clear()
    out = tmp_path / "cal.md"
    assert cli.main(["calibrate", str(PAIRS), "--out", str(out)]) == 0
    report = out.read_text(encoding="utf-8")
    assert "| 0.80 |" in report and "real-cicd" in report and "Three pairs are verbatim" in report
    # A verifier that entails everything accepts every unsupported pair.
    assert "| 0.80 | 100% | 0% |" in report
    assert (
        markdown(read_pairs(PAIRS), [Verdict(Label.NEUTRAL, 0.1)] * 21, "x").count("neutral") == 21
    )
    capsys.readouterr()


def test_the_real_lakehouse_records_load(session: Session) -> None:
    assert session.get(Evidence, "lakehouse-ci-matrix") is not None
    assert read_file(LAKEHOUSE).evidence


# --- claim strength ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bullet", "evidence", "flagged"),
    [
        ("Led the migration to Snowflake.", "Contributed to the migration to Snowflake.", {"led"}),
        (
            "Architected the staff portal's backend.",
            "Wrote Django REST endpoints.",
            {"architected"},
        ),
        ("Owned on-call for the platform.", "Shared the on-call rotation.", {"owned"}),
        ("Built real-time pipelines.", "Wrote nightly Airflow DAGs.", {"real-time"}),
        ("Single-handedly rebuilt the ETL.", "Rebuilt the ETL.", {"sole"}),
        ("Led the platform team.", "Led the platform team of four engineers.", set()),
        ("Leading the data team.", "Led the data team.", set()),
        ("Wrote DAGs that orchestrate loads.", "Wrote DAGs orchestrating loads.", set()),
        ("Documented 20 architecture decisions.", "Documented 20 architecture decisions.", set()),
    ],
)
def test_claim_strength_needs_its_own_evidence(
    bullet: str, evidence: str, flagged: set[str]
) -> None:
    from grounded.verification.strength import escalations

    assert escalations(bullet, [evidence]) == flagged


def test_an_inflated_verb_is_dropped_before_the_model_is_asked(session: Session) -> None:
    verifier = Scripted()
    report = ground(
        session,
        draft(("Led CI for the lakehouse on Ubuntu and Windows.", ["lakehouse-ci-matrix"])),
        candidates(session),
        verifier,
    )
    assert report.dropped[0].reason == "claims 'led', which the cited evidence does not state"
    assert verifier.calls == []


def test_the_full_gate_and_nli_alone_are_both_reported() -> None:
    pairs = read_pairs(PAIRS)
    everything_entailed = [Verdict(Label.ENTAILMENT, 0.99)] * len(pairs.pairs)
    full = {r.threshold: r for r in sweep(pairs, everything_entailed, full_gate=True)}
    alone = {r.threshold: r for r in sweep(pairs, everything_entailed, full_gate=False)}
    assert alone[0.95].false_accept == 1.0  # a verifier that says yes to everything
    # ...still loses the five pairs the deterministic checks stop.
    assert full[0.95].false_accept == pytest.approx(8 / 13)
    report = markdown(pairs, everything_entailed, "yes-man")
    assert "## Full gate" in report and "## NLI alone" in report
    assert "| syn-led | synthetic | no | strength led |" in report
