"""The admin UI: it renders, it writes through ADR-002's rules, and it resists CSRF and XSS."""

from __future__ import annotations

import re
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from grounded.api.app import create_app
from grounded.api.csrf import COOKIE
from grounded.evidence.models import Base


@pytest.fixture
def client() -> Iterator[TestClient]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    with TestClient(create_app(engine), follow_redirects=False) as c:
        yield c


def token(client: TestClient) -> str:
    page = client.get("/admin/evidence/new").text
    match = re.search(r'name="csrf_token" value="([^"]+)"', page)
    assert match
    return match.group(1)


def form(**overrides: str) -> dict[str, str]:
    data = {
        "id": "shipped-thing",
        "kind": "achievement",
        "statement": "Shipped the thing to production.",
    }
    data.update(overrides)
    return data


def test_the_index_renders_and_counts(client: TestClient) -> None:
    page = client.get("/admin")
    assert page.status_code == 200
    assert "unverified (0)" in page.text


def test_the_form_sets_a_csrf_cookie(client: TestClient) -> None:
    t = token(client)
    assert client.cookies.get(COOKIE) == t


def test_a_post_without_a_token_is_refused(client: TestClient) -> None:
    client.get("/admin/evidence/new")
    assert client.post("/admin/evidence", data=form()).status_code == 403


def test_a_post_with_a_forged_token_is_refused(client: TestClient) -> None:
    client.get("/admin/evidence/new")
    r = client.post("/admin/evidence", data={**form(), "csrf_token": "forged"})
    assert r.status_code == 403


def test_a_cross_origin_post_is_refused_even_with_a_token(client: TestClient) -> None:
    t = token(client)
    r = client.post(
        "/admin/evidence",
        data={**form(), "csrf_token": t},
        headers={"origin": "https://evil.example"},
    )
    assert r.status_code == 403


def test_creating_a_record(client: TestClient) -> None:
    t = token(client)
    r = client.post("/admin/evidence", data={**form(), "csrf_token": t})
    assert r.status_code == 303
    assert r.headers["location"] == "/admin/evidence/shipped-thing"
    page = client.get("/admin/evidence/shipped-thing").text
    assert "Shipped the thing to production." in page and "unverified" in page


def test_invalid_input_is_shown_not_stored(client: TestClient) -> None:
    t = token(client)
    r = client.post("/admin/evidence", data={**form(id="Bad_Id"), "csrf_token": t})
    assert r.status_code == 400
    assert "id:" in r.text
    assert "No records" in client.get("/admin").text


def test_a_metric_needs_its_value(client: TestClient) -> None:
    t = token(client)
    r = client.post("/admin/evidence", data={**form(kind="metric"), "csrf_token": t})
    assert r.status_code == 400


def test_editing_through_the_ui_bumps_the_revision_and_unverifies(client: TestClient) -> None:
    t = token(client)
    client.post("/admin/evidence", data={**form(), "csrf_token": t})
    client.post(
        "/admin/evidence/shipped-thing/verify",
        data={"method": "self_attested", "by": "me", "csrf_token": t},
    )
    assert "verified" in client.get("/admin?status=verified").text

    client.post(
        "/admin/evidence",
        data={**form(statement="Led the whole thing to production."), "csrf_token": t},
    )
    page = client.get("/admin/evidence/shipped-thing").text
    assert "Revision 2" in page
    assert 'class="pill unverified"' in page


def test_verify_and_reject_through_the_ui(client: TestClient) -> None:
    t = token(client)
    client.post("/admin/evidence", data={**form(), "csrf_token": t})
    r = client.post(
        "/admin/evidence/shipped-thing/verify",
        data={"method": "third_party", "by": "team lead", "csrf_token": t},
    )
    assert r.status_code == 303
    assert "team lead" in client.get("/admin/evidence/shipped-thing").text

    client.post("/admin/evidence/shipped-thing/reject", data={"by": "reviewer", "csrf_token": t})
    assert 'class="pill rejected"' in client.get("/admin/evidence/shipped-thing").text


def test_verify_and_reject_forms_need_the_token_too(client: TestClient) -> None:
    t = token(client)
    client.post("/admin/evidence", data={**form(), "csrf_token": t})
    r1 = client.post(
        "/admin/evidence/shipped-thing/verify", data={"method": "self_attested", "by": "me"}
    )
    r2 = client.post("/admin/evidence/shipped-thing/reject", data={"by": "me"})
    assert r1.status_code == r2.status_code == 403


def test_bad_verification_input_is_422_and_unknown_records_404(client: TestClient) -> None:
    t = token(client)
    client.post("/admin/evidence", data={**form(), "csrf_token": t})
    bad = client.post(
        "/admin/evidence/shipped-thing/verify",
        data={"method": "telepathy", "by": "me", "csrf_token": t},
    )
    assert bad.status_code == 422
    assert client.get("/admin/evidence/nope").status_code == 404
    assert (
        client.post("/admin/evidence/nope/reject", data={"by": "me", "csrf_token": t}).status_code
        == 404
    )


def test_statements_are_escaped(client: TestClient) -> None:
    """Evidence text is user input; the page must render it, never run it."""
    t = token(client)
    client.post(
        "/admin/evidence",
        data={**form(statement="<script>alert('x')</script> did a thing"), "csrf_token": t},
    )
    page = client.get("/admin").text
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page
