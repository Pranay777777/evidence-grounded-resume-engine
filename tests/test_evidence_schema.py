"""The evidence file format rejects what a person could get wrong."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from grounded.evidence.schema import EvidenceFile, EvidenceIn
from grounded.evidence.store import read_file

FIXTURE = Path(__file__).parent / "fixtures" / "example_evidence.yaml"


def fact(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "a-fact",
        "kind": "achievement",
        "statement": "Did something checkable.",
    }
    base.update(overrides)
    return base


def test_the_fixture_is_valid() -> None:
    data = read_file(FIXTURE)
    assert (len(data.roles), len(data.projects), len(data.evidence)) == (1, 1, 3)


@pytest.mark.parametrize("bad", ["Has-Caps", "under_score", "-leading", "trailing-", "a b"])
def test_ids_must_be_slugs(bad: str) -> None:
    with pytest.raises(ValidationError):
        EvidenceIn.model_validate(fact(id=bad))


@pytest.mark.parametrize("bad", ["2026-13", "2026-6", "26-06", "June 2026"])
def test_months_must_be_yyyy_mm(bad: str) -> None:
    with pytest.raises(ValidationError):
        EvidenceIn.model_validate(fact(month=bad))


def test_a_yaml_date_is_read_as_its_month() -> None:
    assert EvidenceIn.model_validate(fact(month=date(2026, 6, 1))).month == "2026-06"


def test_a_metric_needs_a_value() -> None:
    with pytest.raises(ValidationError, match="metric"):
        EvidenceIn.model_validate(fact(kind="metric"))


def test_only_a_metric_may_carry_a_value() -> None:
    with pytest.raises(ValidationError, match="only metric records"):
        EvidenceIn.model_validate(fact(metric={"value": 3, "unit": "x"}))


def test_artifact_verification_needs_a_link() -> None:
    with pytest.raises(ValidationError, match="artifact_url"):
        EvidenceIn.model_validate(fact(verification={"method": "artifact", "by": "CI"}))


def test_unknown_fields_are_rejected_not_ignored() -> None:
    """A typo like `statment` must fail loudly, not load a record without its fact."""
    with pytest.raises(ValidationError):
        EvidenceIn.model_validate(fact(statment="oops"))


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate evidence id"):
        EvidenceFile.model_validate({"evidence": [fact(), fact()]})


def test_a_role_cannot_end_before_it_starts() -> None:
    role = {"id": "r", "title": "t", "organisation": "o", "start": "2025-06", "end": "2024-01"}
    with pytest.raises(ValidationError, match="ends"):
        EvidenceFile.model_validate({"roles": [role]})


def test_the_hash_tracks_the_fact_not_the_verification() -> None:
    plain = EvidenceIn.model_validate(fact())
    verified = EvidenceIn.model_validate(fact(verification={"method": "self_attested", "by": "me"}))
    edited = EvidenceIn.model_validate(fact(statement="Did something else entirely."))
    assert plain.content_hash() == verified.content_hash()
    assert plain.content_hash() != edited.content_hash()


def test_skills_are_not_part_of_the_fact() -> None:
    assert (
        EvidenceIn.model_validate(fact(skills=["a"])).content_hash()
        == EvidenceIn.model_validate(fact(skills=["b"])).content_hash()
    )


DRAFTS = sorted((Path(__file__).resolve().parents[1] / "evidence" / "drafts").glob("*.yaml"))


@pytest.mark.parametrize("path", DRAFTS, ids=[p.name for p in DRAFTS])
def test_committed_drafts_stay_valid(path: Path) -> None:
    """A draft that stops validating would fail at load time, on someone's machine."""
    data = read_file(path)
    assert data.evidence
    assert all(e.verification is None for e in data.evidence), (
        "drafts are never pre-verified: verifying is the candidate's act"
    )
