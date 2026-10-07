"""Deterministic résumé: every bullet is a verified record's exact statement, cited."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from grounded.evidence.enums import VerificationMethod
from grounded.evidence.models import Base
from grounded.evidence.schema import EvidenceFile
from grounded.evidence.store import load, reject, verify
from grounded.resume.__main__ import find_browser, pdf_pages, print_pdf
from grounded.resume.build import Profile, ResumeError, build, read_profile
from grounded.resume.html import CSS, render

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
                "id": "proj-demo",
                "kind": "achievement",
                "statement": "Deployed proj as a public demo.",
                "project": "proj",
                "artifact_url": "https://proj.example.org",
            },
            {
                "id": "edu-degree",
                "kind": "education",
                "statement": "BSc, Physics · Uni",
            },
            {
                "id": "edu-gpa",
                "kind": "metric",
                "statement": "Holds a CGPA of 8.3 in the BSc.",
                "metric": {"value": 8.3, "unit": "CGPA"},
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
        verify(s, "proj-demo", VerificationMethod.ARTIFACT, "Pranay")
        verify(s, "edu-degree", VerificationMethod.SELF_ATTESTED, "Pranay")
        verify(s, "edu-gpa", VerificationMethod.SELF_ATTESTED, "Pranay")
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
    assert resume.experience[0].subtitle == "Apr 2026 \u2013 Sep 2026"


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


def test_html_escapes_and_links_without_marks(session: Session) -> None:
    resume = build(session, profile())
    html = render(resume)
    assert "<script>" not in html and "&lt;script&gt; &amp; quoting-safe" in html
    assert "Acme &lt;Corp&gt;" in html
    # each bullet links to its proof; record ids stay out of the page
    for ident in ("acme-work", "proj-metric", "proj-feature"):
        assert ident not in html
    assert '<a href="https://github.com/x/proj/actions">The proj suite has 99 tests.</a>' in html
    # self-attested stays in the data, but the page carries no mark for it
    assert resume.experience[0].bullets[0].self_attested
    assert "†" not in html and "<sup>" not in html
    assert "<li>Built an internal tool at Acme.</li>" in html
    assert "Apr 2026 \u2013 Sep 2026" in html
    assert (
        '<div class="footer">Generated from 3 evidence records · project bullets link to '
        "public proof · github.com/Pranay777777/evidence-grounded-resume-engine</div>"
    ) in html
    assert render(resume) == html  # no date or other state: the same input, the same page


def test_education_reads_as_one_line(session: Session) -> None:
    html = render(build(session, profile(education=["edu-degree", "edu-gpa"])))
    assert '<div class="inline"><span>BSc, Physics · Uni</span><span>CGPA 8.3</span></div>' in html
    assert "Holds a CGPA" not in html  # the metric shows as its unit and value


def test_a_demo_link_comes_from_a_record_verified_by_its_url(session: Session) -> None:
    demo = {"project": "proj", "evidence": ["proj-metric"], "demo": "proj-demo"}
    resume = build(session, profile(projects=[demo]))
    assert [b.evidence_id for b in resume.cited] == ["acme-work", "proj-metric", "proj-demo"]
    html = render(resume)
    assert '<a class="demo" href="https://proj.example.org/">live demo</a>' in html
    assert "Deployed proj as a public demo." not in html  # the link, not a bullet
    # a record can be both a bullet and the demo, and is counted once
    both = {"project": "proj", "evidence": ["proj-demo"], "demo": "proj-demo"}
    cited = build(session, profile(projects=[both])).cited
    assert [b.evidence_id for b in cited] == ["acme-work", "proj-demo"]


@pytest.mark.parametrize(
    ("record", "problem"),
    [
        ("proj-feature", "proj-feature: a demo needs a record verified by its URL"),
        ("proj-draft", "proj-draft: unverified, not verified"),
        ("nope", "nope: not in the evidence store"),
    ],
)
def test_a_demo_without_a_url_verified_record_stops_the_build(
    session: Session, record: str, problem: str
) -> None:
    demo = {"project": "proj", "evidence": ["proj-metric"], "demo": record}
    with pytest.raises(ResumeError, match=problem):
        build(session, profile(projects=[demo]))


def test_profile_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text("name: X\nheadline: Y\nsummary: invented prose\n", encoding="utf-8")
    with pytest.raises(ValueError, match="summary"):
        read_profile(path)


def test_find_browser_prefers_an_explicit_path() -> None:
    assert find_browser("/opt/my-chrome") == "/opt/my-chrome"


def test_layout_stays_within_the_agreed_limits() -> None:
    """One page by layout, not by squeezing: margins >= 12 mm, body 9.5-10 pt, lines >= 1.2."""
    margins = re.search(r"@page \{ size: A4; margin: ([\d.]+)mm ([\d.]+)mm; \}", CSS)
    assert margins and min(map(float, margins.groups())) >= 12
    body = re.search(r"body \{ font: ([\d.]+)pt/([\d.]+) ", CSS)
    assert body
    size, line_height = map(float, body.groups())
    assert 9.5 <= size <= 10 and line_height >= 1.2
    name = re.search(r"h1 \{ font-size: ([\d.]+)pt", CSS)
    others = [float(x) for x in re.findall(r"font-size: ([\d.]+)pt", CSS)]
    assert name and float(name.group(1)) == max([*others, size])  # the name stays the biggest


def test_pdf_pages_counts_page_objects(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"<< /Type /Pages /Count 2 >> << /Type /Page >> <</Type/Page/Parent 1 0 R>>")
    assert pdf_pages(pdf) == 2


@pytest.mark.skipif(find_browser(None) is None, reason="no Chromium-family browser")
def test_the_printed_resume_is_one_page(session: Session, tmp_path: Path) -> None:
    html, pdf = tmp_path / "resume.html", tmp_path / "resume.pdf"
    html.write_text(render(build(session, profile())), encoding="utf-8")
    print_pdf(find_browser(None) or "", html, pdf)
    assert pdf_pages(pdf) == 1
