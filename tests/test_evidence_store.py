"""Loading and verifying evidence — above all, that a changed fact loses its verification."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from grounded.evidence.enums import VerificationMethod, VerificationStatus
from grounded.evidence.models import Base, Evidence
from grounded.evidence.schema import EvidenceFile
from grounded.evidence.store import (
    EvidenceError,
    citable,
    load,
    read_file,
    reject,
    verify,
)

FIXTURE = Path(__file__).parent / "fixtures" / "example_evidence.yaml"
NOW = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


@pytest.fixture
def engine() -> Engine:
    eng = create_engine("sqlite://")

    @event.listens_for(eng, "connect")
    def _fk(conn: object, _: object) -> None:
        cur = conn.cursor()  # type: ignore[attr-defined]
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with Session(engine) as s:
        yield s


def example() -> EvidenceFile:
    return read_file(FIXTURE)


def edited(statement: str) -> EvidenceFile:
    raw = yaml.safe_load(FIXTURE.read_text())
    raw["evidence"][1]["statement"] = statement
    return EvidenceFile.model_validate(raw)


def test_a_first_load_creates_everything(session: Session) -> None:
    report = load(session, example(), NOW)
    assert sorted(report.created) == [
        "example-oncall",
        "example-pipeline-built",
        "example-pipeline-latency",
    ]
    assert report.verified == ["example-pipeline-latency"]


def test_new_records_are_unverified_unless_the_file_verifies_them(session: Session) -> None:
    load(session, example(), NOW)
    built = session.get(Evidence, "example-pipeline-built")
    latency = session.get(Evidence, "example-pipeline-latency")
    assert built is not None and latency is not None
    assert built.verification_status == VerificationStatus.UNVERIFIED
    assert latency.verification_status == VerificationStatus.VERIFIED
    assert latency.verification_method == VerificationMethod.ARTIFACT
    assert latency.verified_by == "CI dashboard"


def test_loading_twice_changes_nothing(session: Session) -> None:
    load(session, example(), NOW)
    report = load(session, example(), NOW)
    assert report.created == [] and report.revised == []
    assert len(report.unchanged) == 3


def test_a_changed_fact_is_an_unverified_fact(session: Session) -> None:
    """The rule the whole store exists for (ADR-002)."""
    load(session, example(), NOW)
    raw = yaml.safe_load(FIXTURE.read_text())
    raw["evidence"][1]["statement"] = "The nightly pipeline completes in under two minutes."
    del raw["evidence"][1]["verification"]

    report = load(session, EvidenceFile.model_validate(raw), NOW)

    record = session.get(Evidence, "example-pipeline-latency")
    assert record is not None
    assert report.revised == ["example-pipeline-latency"]
    assert record.revision == 2
    assert record.verification_status == VerificationStatus.UNVERIFIED
    assert record.verification_method is None and record.verified_by is None
    assert not record.citable


def test_an_edit_can_be_re_verified_in_the_same_file(session: Session) -> None:
    load(session, example(), NOW)
    report = load(session, edited("The nightly pipeline completes in under eight minutes."), NOW)
    record = session.get(Evidence, "example-pipeline-latency")
    assert record is not None
    assert report.revised == report.verified == ["example-pipeline-latency"]
    assert record.revision == 2 and record.citable


def test_retagging_skills_is_not_a_revision(session: Session) -> None:
    load(session, example(), NOW)
    raw = yaml.safe_load(FIXTURE.read_text())
    raw["evidence"][0]["skills"] = ["python", "sql", "airflow"]
    report = load(session, EvidenceFile.model_validate(raw), NOW)
    record = session.get(Evidence, "example-pipeline-built")
    assert record is not None
    assert report.revised == [] and record.revision == 1
    assert record.skills == ["python", "sql", "airflow"]


def test_a_file_can_verify_an_already_loaded_record(session: Session) -> None:
    load(session, example(), NOW)
    raw = yaml.safe_load(FIXTURE.read_text())
    raw["evidence"][0]["verification"] = {"method": "self_attested", "by": "the candidate"}
    report = load(session, EvidenceFile.model_validate(raw), NOW)
    assert report.verified == ["example-pipeline-built"]


def test_only_verified_records_are_citable(session: Session) -> None:
    load(session, example(), NOW)
    assert [e.id for e in citable(session)] == ["example-pipeline-latency"]


def test_verify_records_how_and_by_whom(session: Session) -> None:
    load(session, example(), NOW)
    record = verify(session, "example-oncall", VerificationMethod.THIRD_PARTY, "team lead", NOW)
    assert record.citable and record.verified_by == "team lead"


def test_artifact_verification_needs_a_link(session: Session) -> None:
    load(session, example(), NOW)
    with pytest.raises(EvidenceError, match="artifact_url"):
        verify(session, "example-oncall", VerificationMethod.ARTIFACT, "me", NOW)


def test_a_rejected_record_is_kept_but_never_citable(session: Session) -> None:
    load(session, example(), NOW)
    record = reject(session, "example-pipeline-latency", "reviewer", NOW)
    assert record.verification_status == VerificationStatus.REJECTED
    assert session.get(Evidence, "example-pipeline-latency") is not None
    assert citable(session) == []


def test_verifying_or_rejecting_an_unknown_id_is_an_error(session: Session) -> None:
    with pytest.raises(EvidenceError, match="no evidence"):
        verify(session, "missing", VerificationMethod.SELF_ATTESTED, "me")
    with pytest.raises(EvidenceError, match="no evidence"):
        reject(session, "missing", "me")


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            {
                "evidence": [
                    {
                        "id": "e",
                        "kind": "skill",
                        "statement": "Used a thing daily.",
                        "project": "nope",
                    }
                ]
            },
            "unknown project",
        ),
        (
            {
                "evidence": [
                    {"id": "e", "kind": "skill", "statement": "Used a thing daily.", "role": "nope"}
                ]
            },
            "unknown role",
        ),
        ({"projects": [{"id": "p", "name": "P", "role": "nope"}]}, "unknown role"),
    ],
)
def test_references_must_resolve(session: Session, raw: dict[str, object], message: str) -> None:
    with pytest.raises(EvidenceError, match=message):
        load(session, EvidenceFile.model_validate(raw), NOW)


def test_a_failed_load_writes_nothing(session: Session) -> None:
    bad = {
        "projects": [{"id": "p", "name": "P"}],
        "evidence": [
            {"id": "e", "kind": "skill", "statement": "Used a thing daily.", "project": "missing"}
        ],
    }
    with pytest.raises(EvidenceError):
        load(session, EvidenceFile.model_validate(bad), NOW)
    assert session.execute(text("SELECT count(*) FROM project")).scalar_one() == 0


def test_a_load_never_deletes(session: Session) -> None:
    load(session, example(), NOW)
    load(session, EvidenceFile.model_validate({}), NOW)
    assert session.execute(text("SELECT count(*) FROM evidence")).scalar_one() == 3


def test_the_database_refuses_a_metric_without_a_value(session: Session) -> None:
    """The constraints are a second line behind validation, not decoration."""
    session.add(
        Evidence(
            id="raw",
            kind="metric",
            statement="A number, supposedly.",
            content_hash="x",
            revision=1,
            verification_status="unverified",
            skills=[],
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()


def test_the_database_refuses_verified_without_a_method(session: Session) -> None:
    session.add(
        Evidence(
            id="raw",
            kind="skill",
            statement="Used a thing daily.",
            content_hash="x",
            revision=1,
            verification_status="verified",
            skills=[],
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
