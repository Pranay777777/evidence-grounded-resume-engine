"""Auth and multi-tenancy (step 60): tenant A cannot read, write or retrieve tenant B's data."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from grounded.api import drafts
from grounded.api.__main__ import main as api_main
from grounded.api.app import create_app
from grounded.api.auth import AuthError, issue, verify
from grounded.api.guards import digest, parse_keys
from grounded.config import Settings, get_settings
from grounded.evidence.models import Base, Evidence
from grounded.evidence.schema import EvidenceIn
from grounded.evidence.store import EvidenceError, load, read_file, upsert_evidence
from grounded.evidence.tenancy import CrossTenantWriteError, scope
from grounded.generation.llm import ChatResponse
from grounded.generation.service import retrieve
from grounded.retrieval.embedding import get_embedder
from grounded.retrieval.index import embed_pending
from grounded.verification.nli import Label, Verdict

SECRET = "s" * 48
FIXTURE = Path(__file__).parent / "fixtures" / "retrieval_corpus.yaml"


def settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {"auth": "jwt", "jwt_secret": SecretStr(SECRET), **overrides}
    return Settings(**values)


def record(
    ident: str, statement: str = "Built the evidence store for tenant tests."
) -> dict[str, Any]:
    return {"id": ident, "kind": "achievement", "statement": statement}


# --- the ORM layer --------------------------------------------------------------


@pytest.fixture
def engine() -> Any:
    e = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(e)
    return e


def test_sessions_only_see_and_stamp_their_own_tenant(engine: Any) -> None:
    with scope(Session(engine), "acme") as s:
        upsert_evidence(s, EvidenceIn.model_validate(record("acme-fact")))
        s.commit()
    with scope(Session(engine), "globex") as s:
        upsert_evidence(s, EvidenceIn.model_validate(record("globex-fact")))
        s.commit()
        assert s.get(Evidence, "acme-fact") is None
        assert [r.id for r in s.scalars(select(Evidence))] == ["globex-fact"]
        with pytest.raises(EvidenceError, match="already in use"):
            upsert_evidence(s, EvidenceIn.model_validate(record("acme-fact")))
    with Session(engine) as unscoped:  # the CLI's bulk tools see every tenant
        rows = {(r.id, r.tenant_id) for r in unscoped.scalars(select(Evidence))}
        assert rows == {("acme-fact", "acme"), ("globex-fact", "globex")}


def test_relationship_loads_are_filtered_too(engine: Any) -> None:
    with scope(Session(engine), "acme") as s:
        load(s, read_file(FIXTURE))
        s.commit()
    with scope(Session(engine), "globex") as s:
        assert s.scalars(select(Evidence)).all() == []
    with scope(Session(engine), "acme") as s:
        loaded = s.scalars(select(Evidence).where(Evidence.project_id.is_not(None))).first()
        assert loaded is not None and loaded.project is not None
        assert loaded.project.tenant_id == "acme"


def test_a_session_cannot_write_another_tenants_row(engine: Any) -> None:
    with scope(Session(engine), "acme") as s:
        s.add(
            Evidence(
                id="x",
                kind="achievement",
                statement="Did a thing for a test.",
                content_hash="h",
                tenant_id="globex",
            )
        )
        with pytest.raises(CrossTenantWriteError):
            s.flush()


def test_retrieval_for_one_tenant_never_returns_anothers_records(engine: Any) -> None:
    with scope(Session(engine), "acme") as s:
        load(s, read_file(FIXTURE))
        embed_pending(s, get_embedder("hashing"))
        s.commit()
    config = Settings(embedder="hashing")
    with scope(Session(engine), "globex") as s:
        assert retrieve(s, "Delta Lake incremental loads", config, 5, False) == []
    with scope(Session(engine), "acme") as s:
        assert retrieve(s, "Delta Lake incremental loads", config, 5, False)


# --- tokens ---------------------------------------------------------------------


def test_tokens_round_trip_and_reject_tampering() -> None:
    config = settings()
    token = issue(config, "ana", "acme", {"read_evidence", "generate"}, budget=900)
    principal = verify(token, config)
    assert (principal.subject, principal.tenant, principal.daily_tokens) == ("ana", "acme", 900)
    assert principal.scopes == {"read_evidence", "generate"} and principal.name == "acme/ana"
    assert verify(issue(config, "bo", "acme", set()), config).daily_tokens == 50_000

    with pytest.raises(AuthError, match="invalid token"):
        verify(token, settings(jwt_secret=SecretStr("t" * 48)))
    old = issue(config, "ana", "acme", {"admin"}, now=datetime.now(UTC) - timedelta(hours=2))
    with pytest.raises(AuthError, match="expired"):
        verify(old, config)
    unsigned = jwt.encode({"sub": "x", "tenant": "acme", "scope": "admin"}, None, algorithm="none")
    with pytest.raises(AuthError):
        verify(unsigned, config)
    wrong_audience = issue(settings(jwt_audience="other"), "ana", "acme", {"admin"})
    with pytest.raises(AuthError, match=r"(?i)audience"):
        verify(wrong_audience, config)
    no_tenant = jwt.encode(
        {
            "sub": "x",
            "scope": "admin",
            "iss": "grounded",
            "aud": "grounded-api",
            "iat": datetime.now(UTC),
            "exp": datetime.now(UTC) + timedelta(hours=1),
        },
        SECRET,
        algorithm="HS256",
    )
    with pytest.raises(AuthError, match="tenant"):
        verify(no_tenant, config)


def test_issuing_validates_scopes_tenants_and_the_secret() -> None:
    with pytest.raises(AuthError, match="unknown scope"):
        issue(settings(), "a", "acme", {"root"})
    with pytest.raises(AuthError, match="tenant 'Acme Corp'"):
        issue(settings(), "a", "Acme Corp", {"admin"})
    with pytest.raises(AuthError, match="at least 32 bytes"):
        issue(settings(jwt_secret=SecretStr("short")), "a", "acme", {"admin"})


# --- the API --------------------------------------------------------------------


class Entailed:
    name = "fake-nli"

    def check(self, premise: str, hypothesis: str) -> Verdict:
        return Verdict(Label.ENTAILMENT, 0.99)


class FakeClient:
    model = "fake/model"

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        args = {
            "bullets": [
                {
                    "text": "Implemented incremental loads with Delta Lake MERGE.",
                    "evidence_ids": ["ex-delta-merge"],
                }
            ]
        }
        return ChatResponse(
            None, json.dumps(args), self.model, usage={"prompt_tokens": 10, "completion_tokens": 5}
        )


@pytest.fixture
def api(engine: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("AUTH", "jwt")
    monkeypatch.setenv("JWT_SECRET", SECRET)
    monkeypatch.setenv("EMBEDDER", "hashing")
    monkeypatch.setenv("RATE_LIMIT", "100/minute")
    monkeypatch.setenv("API_KEYS", f"acme-bot={digest('gr_acme')}:1000:acme")
    monkeypatch.setenv("SEMANTIC_CACHE_PATH", str(tmp_path / "cache.jsonl"))
    get_settings.cache_clear()
    drafts.limiter.reset()
    with scope(Session(engine), "acme") as s:
        load(s, read_file(FIXTURE))
        embed_pending(s, get_embedder("hashing"))
        s.commit()
    app = create_app(
        engine, client_factory=lambda spec, config: FakeClient(), verifier=lambda: Entailed()
    )
    with TestClient(app) as client:
        yield client
    get_settings.cache_clear()


def bearer(tenant: str, *scopes: str, subject: str = "user") -> dict[str, str]:
    return {"Authorization": f"Bearer {issue(get_settings(), subject, tenant, set(scopes))}"}


def test_tenant_a_cannot_read_tenant_bs_evidence(api: TestClient) -> None:
    """The proof the plan asks for, through the HTTP API."""
    acme = bearer("acme", "read_evidence", "admin")
    globex = bearer("globex", "read_evidence", "admin")

    assert api.get("/evidence/ex-delta-merge", headers=acme).status_code == 200
    assert api.get("/evidence/ex-delta-merge", headers=globex).status_code == 404
    assert api.get("/evidence", headers=globex).json() == []
    assert api.get("/projects", headers=globex).json() == []
    assert api.get("/roles", headers=globex).json() == []
    assert len(api.get("/evidence", headers=acme).json()) > 1

    body = {"method": "self_attested", "by": "mallory"}
    assert api.post("/evidence/ex-delta-merge/verify", json=body, headers=globex).status_code == 404
    assert (
        api.post("/evidence/ex-delta-merge/reject", json={"by": "m"}, headers=globex).status_code
        == 404
    )
    taken = api.put("/evidence/ex-delta-merge", json=record("ex-delta-merge"), headers=globex)
    assert taken.status_code == 422 and "already in use" in taken.json()["detail"]
    mine = api.put("/evidence/globex-own", json=record("globex-own"), headers=globex)
    assert mine.status_code == 200
    assert api.get("/evidence/globex-own", headers=acme).status_code == 404


def test_scopes_are_enforced(api: TestClient) -> None:
    reader = bearer("acme", "read_evidence")
    assert api.get("/evidence", headers=reader).status_code == 200
    denied = api.put("/evidence/new-one", json=record("new-one"), headers=reader)
    assert denied.status_code == 403 and "'admin' scope" in denied.json()["detail"]
    assert (
        api.post(
            "/v1/drafts",
            json={"job_description": "Data engineer with Delta Lake loads."},
            headers=reader,
        ).status_code
        == 403
    )
    assert api.get("/evidence").status_code == 401
    assert api.get("/evidence", headers={"Authorization": "Bearer nonsense"}).status_code == 401
    assert api.get("/admin").status_code == 404  # local tools are not served with AUTH=jwt


def test_drafts_use_only_the_callers_tenant(api: TestClient) -> None:
    jd = {"job_description": "Data engineer with Delta Lake incremental loads."}
    acme = api.post("/v1/drafts", json=jd, headers=bearer("acme", "generate"))
    assert acme.status_code == 200 and acme.json()["kept"]
    globex = api.post("/v1/drafts", json=jd, headers=bearer("globex", "generate"))
    assert globex.status_code == 422  # acme's evidence is invisible; no model call is made
    assert "no verified evidence" in globex.json()["detail"]
    assert api.app.state.breaker.failures == 0  # type: ignore[attr-defined]
    usage = api.get("/v1/usage", headers=bearer("acme", "generate")).json()
    assert usage["tenant"] == "acme" and usage["used_today"] == 15


def test_api_keys_carry_a_tenant(api: TestClient) -> None:
    key = {"X-API-Key": "gr_acme"}
    assert api.get("/v1/usage", headers=key).json()["tenant"] == "acme"
    assert api.get("/evidence", headers=key).status_code == 403  # keys can only generate
    assert api.get("/v1/usage", headers={"X-API-Key": "gr_wrong"}).status_code == 401
    assert parse_keys(f"k={digest('x')}:5")[0].tenant == "default"
    with pytest.raises(ValueError, match=r"name=sha256:budget\[:tenant\]"):
        parse_keys(f"k={digest('x')}:5:acme:extra")


def test_bearer_tokens_need_jwt_mode(monkeypatch: pytest.MonkeyPatch, engine: Any) -> None:
    monkeypatch.setenv("JWT_SECRET", SECRET)
    get_settings.cache_clear()
    with TestClient(create_app(engine)) as client:
        token = issue(get_settings(), "a", "acme", {"read_evidence"})
        refused = client.get("/evidence", headers={"Authorization": f"Bearer {token}"})
        assert refused.status_code == 401 and "AUTH=jwt" in refused.json()["detail"]
        assert client.get("/evidence").status_code == 200  # AUTH=off: local, default tenant
    get_settings.cache_clear()


def test_jwt_mode_refuses_to_start_without_a_strong_secret(
    monkeypatch: pytest.MonkeyPatch, engine: Any
) -> None:
    monkeypatch.setenv("AUTH", "jwt")
    monkeypatch.setenv("JWT_SECRET", "weak")
    get_settings.cache_clear()
    with pytest.raises(AuthError, match="at least 32 bytes"):
        create_app(engine)
    get_settings.cache_clear()


def test_token_and_keygen_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("JWT_SECRET", SECRET)
    get_settings.cache_clear()
    assert (
        api_main(
            [
                "token",
                "--subject",
                "ana",
                "--tenant",
                "acme",
                "--scopes",
                "generate, read_evidence",
                "--budget",
                "100",
            ]
        )
        == 0
    )
    token = capsys.readouterr().out.strip()
    principal = verify(token, get_settings())
    assert principal.tenant == "acme" and principal.daily_tokens == 100
    assert api_main(["token", "--subject", "a", "--tenant", "acme", "--scopes", "root"]) == 1
    assert "unknown scope" in capsys.readouterr().err
    assert api_main(["keygen", "bot", "--tenant", "acme"]) == 0
    entry = capsys.readouterr().out.split("in .env:")[1].strip()
    assert entry.endswith(":50000:acme") and parse_keys(entry)[0].tenant == "acme"
    get_settings.cache_clear()
