"""Deterministic résumé: every bullet is a verified record's exact statement, cited."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from grounded.evidence.enums import VerificationMethod
from grounded.evidence.models import Base
from grounded.evidence.schema import EvidenceFile
from grounded.evidence.store import load, reject, verify
from grounded.resume.__main__ import find_browser
from grounded.resume.build import Profile, ResumeError, build, read_profile
from grounded.resume.html import render

STORE = EvidenceFile.model_validate(
    {
        "roles": [
            {
                "id": "acme",
                "title": "Data Engineer",
                "organisation": "Acme <Corp>",
                "start": "2026-04",
                "end": "2026-09",
            }
        ],
        "projects": [
            {
                "id": "proj",
                "name": "proj",
                "summary": "A thing.",
                "repo_url": "https://github.com/x/proj",
            }
        ],
        "evidence": [
            {
                "id": "proj-metric",
                "kind": "metric",
                "statement": "The proj suite has 99 tests.",
                "project": "proj",
                "metric": {"value": 99, "unit": "tests"},
                "artifact_url": "https://github.com/x/proj/actions",
            },
            {
                "id": "proj-feature",
                "kind": "achievement",
                "statement": "Built the <script> & quoting-safe exporter for proj.",
                "project": "proj",
            },
            {
                "id": "acme-work",
                "kind": "achievement",
                "statement": "Built an internal tool at Acme.",
                "role": "acme",
            },
            {
                "id": "proj-draft",
                "kind": "achievement",
                "statement": "Claimed something unchecked.",
                "project": "proj",
            },
            {
                "id": "proj-rejected",
                "kind": "achievement",
                "statement": "Claimed something wrong.",
                "project": "proj",
            },
        ],
    }
)


@pytest.fixture
def session() -> Iterator[Session]:
    eng = create_engine("sqlite://")

    @event.listens_for(eng, "connect")
    def _fk(conn: object, _: object) -> None:
        cur = conn.cursor()  # type: ignore[attr-defined]
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(eng)
    with Session(eng) as s:
        load(s, STORE)
        verify(s, "proj-metric", VerificationMethod.ARTIFACT, "Pranay")
        verify(s, "proj-feature", VerificationMethod.THIRD_PARTY, "A reviewer")
        verify(s, "acme-work", VerificationMethod.SELF_ATTESTED, "Pranay")
        reject(s, "proj-rejected", "Pranay")
        yield s


def profile(**overrides: object) -> Profile:
    base: dict[str, object] = {
        "name": "Test Person",
        "headline": "Engineer",
        "contact": [{"label": "github.com/x", "url": "https://github.com/x"}, {"label": "Earth"}],
        "experience": [{"role": "acme", "evidence": ["acme-work"]}],
        "projects": [{"project": "proj", "evidence": ["proj-metric", "proj-feature"]}],
        "skills": [{"group": "Engineering", "items": ["Python", "SQL"]}],
    }
    return Profile.model_validate(base | overrides)


def test_bullets_are_exact_statements_with_citations(session: Session) -> None:
    resume = build(session, profile())
    assert [b.evidence_id for b in resume.cited] == ["acme-work", "proj-metric", "proj-feature"]
    proj = resume.projects[0]
    assert [b.text for b in proj.bullets] == [
        "The proj suite has 99 tests.",
        "Built the <script> & quoting-safe exporter for proj.",
    ]
    assert proj.bullets[0].link == "https://github.com/x/proj/actions"  # its artifact
    assert proj.bullets[1].link == "https://github.com/x/proj"  # no artifact: the repo
    assert resume.experience[0].subtitle == "2026-04 \u2013 2026-09"


@pytest.mark.parametrize(
    ("ids", "problem"),
    [
        (["proj-draft"], "proj-draft: unverified, not verified"),
        (["proj-rejected"], "proj-rejected: rejected, not verified"),
        (["nope"], "nope: not in the evidence store"),
        (["proj-metric", "proj-metric"], "proj-metric: selected twice"),
    ],
)
def test_anything_not_verified_stops_the_build(
    session: Session, ids: list[str], problem: str
) -> None:
    with pytest.raises(ResumeError, match=problem):
        build(session, profile(projects=[{"project": "proj", "evidence": ids}]))


def test_unknown_role_or_project_stops_the_build(session: Session) -> None:
    with pytest.raises(ResumeError, match="role ghost: not in the evidence store"):
        build(session, profile(experience=[{"role": "ghost", "evidence": ["acme-work"]}]))


def test_html_escapes_cites_and_marks_self_attested(session: Session) -> None:
    html = render(build(session, profile()), today=date(2026, 10, 2))
    assert "<script>" not in html and "&lt;script&gt; &amp; quoting-safe" in html
    assert "Acme &lt;Corp&gt;" in html
    for ident in ("acme-work", "proj-metric", "proj-feature"):
        assert f"[{ident}]" in html
    assert 'href="https://github.com/x/proj/actions"' in html
    # only the self-attested record carries the dagger, and the footer explains it
    assert html.count("<sup>†</sup>") == 1
    assert "Built an internal tool at Acme.<sup>†</sup>" in html
    assert "Generated 2026-10-02 from 3 verified evidence records" in html
    assert "† self-attested" in html


def test_profile_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text("name: X\nheadline: Y\nsummary: invented prose\n", encoding="utf-8")
    with pytest.raises(ValueError, match="summary"):
        read_profile(path)


def test_find_browser_prefers_an_explicit_path() -> None:
    assert find_browser("/opt/my-chrome") == "/opt/my-chrome"
