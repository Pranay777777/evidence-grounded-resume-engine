"""The JSON API — and that it cannot be used to sidestep ADR-002."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from grounded.api.app import create_app
from grounded.evidence.models import Base


@pytest.fixture
def client() -> Iterator[TestClient]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    with TestClient(create_app(engine)) as c:
        yield c


def record(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "shipped-thing",
        "kind": "achievement",
        "statement": "Shipped the thing to production.",
    }
    body.update(overrides)
    return body


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_put_creates_an_unverified_record(client: TestClient) -> None:
    r = client.put("/evidence/shipped-thing", json=record())
    assert r.status_code == 200
    body = r.json()
    assert body["outcome"] == "created"
    assert body["record"]["verification_status"] == "unverified"
    assert body["record"]["citable"] is False


def test_putting_the_same_record_again_changes_nothing(client: TestClient) -> None:
    client.put("/evidence/shipped-thing", json=record())
    assert client.put("/evidence/shipped-thing", json=record()).json()["outcome"] == "unchanged"


def test_the_api_cannot_keep_an_edited_fact_verified(client: TestClient) -> None:
    """ADR-002 through the API: same write path, same rule."""
    client.put("/evidence/shipped-thing", json=record())
    client.post("/evidence/shipped-thing/verify", json={"method": "self_attested", "by": "me"})
    r = client.put(
        "/evidence/shipped-thing", json=record(statement="Led the whole thing to production.")
    )
    body = r.json()
    assert body["outcome"] == "revised"
    assert body["record"]["revision"] == 2
    assert body["record"]["verification_status"] == "unverified"
    assert body["record"]["verified_by"] is None


def test_path_and_body_ids_must_agree(client: TestClient) -> None:
    r = client.put("/evidence/other-id", json=record())
    assert r.status_code == 422
    assert "does not match" in r.json()["detail"]


def test_invalid_records_are_refused_with_reasons(client: TestClient) -> None:
    r = client.put("/evidence/Bad_Id", json=record(id="Bad_Id"))
    assert r.status_code == 422


def test_unknown_references_are_refused(client: TestClient) -> None:
    r = client.put("/evidence/shipped-thing", json=record(project="nowhere"))
    assert r.status_code == 422
    assert "unknown project" in r.json()["detail"]


def test_list_filters(client: TestClient) -> None:
    client.put("/projects/p", json={"id": "p", "name": "P"})
    client.put("/evidence/a-one", json=record(id="a-one", project="p"))
    client.put(
        "/evidence/b-two",
        json=record(id="b-two", kind="metric", metric={"value": 3, "unit": "x"}),
    )
    client.post("/evidence/b-two/verify", json={"method": "self_attested", "by": "me"})

    def ids(query: str) -> list[str]:
        return [e["id"] for e in client.get(f"/evidence{query}").json()]

    assert ids("") == ["a-one", "b-two"]
    assert ids("?status=verified") == ["b-two"]
    assert ids("?kind=metric") == ["b-two"]
    assert ids("?project=p") == ["a-one"]


def test_unknown_records_are_404(client: TestClient) -> None:
    assert client.get("/evidence/nope").status_code == 404
    assert (
        client.post(
            "/evidence/nope/verify", json={"method": "self_attested", "by": "me"}
        ).status_code
        == 404
    )
    assert client.post("/evidence/nope/reject", json={"by": "me"}).status_code == 404


def test_artifact_verification_needs_a_link(client: TestClient) -> None:
    client.put("/evidence/shipped-thing", json=record())
    r = client.post("/evidence/shipped-thing/verify", json={"method": "artifact", "by": "CI"})
    assert r.status_code == 422


def test_reject_keeps_the_record(client: TestClient) -> None:
    client.put("/evidence/shipped-thing", json=record())
    r = client.post("/evidence/shipped-thing/reject", json={"by": "reviewer"})
    assert r.json()["verification_status"] == "rejected"
    assert client.get("/evidence/shipped-thing").status_code == 200


def test_there_is_no_delete(client: TestClient) -> None:
    """Records are retired by rejection, never erased (ADR-002)."""
    client.put("/evidence/shipped-thing", json=record())
    assert client.delete("/evidence/shipped-thing").status_code == 405


def test_roles_and_projects(client: TestClient) -> None:
    role = {"id": "r1", "title": "Engineer", "organisation": "Example Corp", "start": "2024-01"}
    assert client.put("/roles/r1", json=role).status_code == 200
    assert (
        client.put("/projects/p1", json={"id": "p1", "name": "P", "role": "r1"}).status_code == 200
    )
    assert [r["id"] for r in client.get("/roles").json()] == ["r1"]
    assert client.get("/projects").json()[0]["role_id"] == "r1"


def test_a_project_needs_a_real_role(client: TestClient) -> None:
    r = client.put("/projects/p1", json={"id": "p1", "name": "P", "role": "ghost"})
    assert r.status_code == 422


def test_role_and_project_ids_must_agree(client: TestClient) -> None:
    role = {"id": "r1", "title": "E", "organisation": "O"}
    assert client.put("/roles/other", json=role).status_code == 422
    assert client.put("/projects/other", json={"id": "p1", "name": "P"}).status_code == 422


def test_the_api_documents_itself(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "/evidence/{evidence_id}" in paths and "/evidence/{evidence_id}/verify" in paths


def test_json_writes_refuse_simple_cross_site_content_types(client: TestClient) -> None:
    """Why the JSON API needs no CSRF token (ADR-003).

    A cross-site page can send a "simple" request — text/plain or a form —
    without a CORS preflight, and could put JSON in the body. FastAPI only
    parses bodies declared as application/json, so those forgeries fail;
    and a real application/json request from another origin triggers a
    preflight this server never grants.
    """
    client.put("/evidence/shipped-thing", json=record())
    forged = client.post(
        "/evidence/shipped-thing/verify",
        content='{"method": "self_attested", "by": "attacker"}',
        headers={"content-type": "text/plain"},
    )
    assert forged.status_code == 422
    assert client.get("/evidence/shipped-thing").json()["verification_status"] == "unverified"


def test_no_cors_is_granted(client: TestClient) -> None:
    preflight = client.options(
        "/evidence/shipped-thing/verify",
        headers={"origin": "https://evil.example", "access-control-request-method": "POST"},
    )
    assert "access-control-allow-origin" not in preflight.headers
