"""The /v1 generation API: keys, budgets, rate limits, circuit breaker (step 59)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from grounded.api import drafts
from grounded.api.__main__ import main as api_main
from grounded.api.app import create_app
from grounded.api.guards import (
    ApiKey,
    CircuitBreaker,
    CircuitOpenError,
    Ledger,
    authenticate,
    digest,
    parse_keys,
)
from grounded.config import get_settings
from grounded.evidence.models import Base
from grounded.evidence.store import load, read_file
from grounded.generation.llm import ChatResponse, LLMError
from grounded.retrieval.embedding import get_embedder
from grounded.retrieval.index import embed_pending
from grounded.verification.nli import Label, Verdict

FIXTURE = Path(__file__).parent / "fixtures" / "retrieval_corpus.yaml"
KEY = "gr_test_key"
JD = {"job_description": "Data engineer with Delta Lake incremental loads."}


class Entailed:
    name = "fake-nli"

    def check(self, premise: str, hypothesis: str) -> Verdict:
        return Verdict(Label.ENTAILMENT, 0.99)


class FakeClient:
    model = "fake/model"

    def __init__(self, text: str = "Implemented incremental loads with Delta Lake MERGE.") -> None:
        self.text = text

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        args = {"bullets": [{"text": self.text, "evidence_ids": ["ex-delta-merge"]}]}
        return ChatResponse(
            None,
            json.dumps(args),
            self.model,
            usage={"prompt_tokens": 400, "completion_tokens": 100},
        )


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("EMBEDDER", "hashing")
    monkeypatch.setenv("API_KEYS", f"tester={digest(KEY)}:1200, other={digest('gr_other')}:10")
    monkeypatch.setenv("RATE_LIMIT", "100/minute")
    monkeypatch.setenv("SEMANTIC_CACHE_PATH", str(tmp_path / "cache.jsonl"))
    get_settings.cache_clear()
    drafts.limiter.reset()
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        load(session, read_file(FIXTURE))
        embed_pending(session, get_embedder("hashing"))
        session.commit()
    app = create_app(
        engine, client_factory=lambda spec, settings: FakeClient(), verifier=lambda: Entailed()
    )
    with TestClient(app) as client:
        yield client
    get_settings.cache_clear()


def post(api: TestClient, key: str | None = KEY, **body: Any) -> Any:
    headers = {"X-API-Key": key} if key else {}
    return api.post("/v1/drafts", json={**JD, **body}, headers=headers)


def test_health_and_prompts_need_no_key(api: TestClient) -> None:
    assert api.get("/v1/health").json() == {"status": "ok", "provider_circuit": "closed"}
    versions = {p["version"]: p["default"] for p in api.get("/v1/prompts").json()}
    assert versions == {"generate-v1": "false", "generate-v2": "true"}
    assert "/v1/drafts" in api.get("/openapi.json").json()["paths"]


def test_drafts_need_a_known_key(api: TestClient) -> None:
    assert post(api, key=None).status_code == 401
    assert post(api, key="gr_wrong").status_code == 401
    assert api.get("/v1/usage").status_code == 401


def test_a_draft_is_gated_and_charged(api: TestClient) -> None:
    response = post(api)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kept"][0]["evidence_ids"] == ["ex-delta-merge"]
    assert body["kept"][0]["entailment"] == 0.99 and body["dropped"] == []
    assert (body["tokens"], body["cache_hit"], body["prompt_version"]) == (
        500,
        False,
        "generate-v2",
    )
    usage = api.get("/v1/usage", headers={"X-API-Key": KEY}).json()
    assert usage == {
        "key": "tester",
        "daily_tokens": 1200,
        "used_today": 500,
        "remaining_today": 700,
    }


def test_a_cache_hit_costs_nothing(api: TestClient) -> None:
    post(api, cache=True)
    again = post(api, cache=True).json()
    assert again["cache_hit"] is True and again["tokens"] == 0


def test_the_budget_stops_calls_before_the_model(api: TestClient) -> None:
    assert post(api).status_code == 200 and post(api).status_code == 200  # 1,000 of 1,200
    assert post(api).status_code == 200  # still 200 left before this call
    blocked = post(api)
    assert blocked.status_code == 429 and "1,200 used up" in blocked.json()["detail"]
    assert post(api, key="gr_other").status_code == 200  # budgets are per key


def test_rate_limit(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RATE_LIMIT", "2/minute")
    get_settings.cache_clear()
    assert [post(api).status_code for _ in range(3)] == [200, 200, 429]


def test_bad_requests(api: TestClient) -> None:
    assert post(api, prompt_version="v9").status_code == 422
    assert (
        api.post(
            "/v1/drafts", json={"job_description": "short"}, headers={"X-API-Key": KEY}
        ).status_code
        == 422
    )


def test_the_circuit_opens_after_provider_failures_and_recovers(api: TestClient) -> None:
    now = [0.0]
    breaker = CircuitBreaker(failures_to_open=2, cooldown_s=30, clock=lambda: now[0])
    app = api.app
    app.state.breaker = breaker  # type: ignore[attr-defined]
    calls = {"n": 0}

    def flaky(spec: str, settings: Any) -> Any:
        calls["n"] += 1
        raise LLMError("provider returned 503: down")

    app.state.client_factory = flaky  # type: ignore[attr-defined]
    assert [post(api).status_code for _ in range(2)] == [502, 502]
    fast = post(api)
    assert fast.status_code == 503 and fast.headers["retry-after"] == "30"
    assert calls["n"] == 2  # the open circuit did not call the provider
    assert api.get("/v1/health").json()["provider_circuit"] == "open"

    now[0] = 31.0
    assert breaker.state == "half-open"
    app.state.client_factory = lambda spec, settings: FakeClient()  # type: ignore[attr-defined]
    assert post(api).status_code == 200
    assert breaker.state == "closed"


def test_unusable_output_does_not_trip_the_circuit(api: TestClient) -> None:
    class Garbage(FakeClient):
        def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
            return ChatResponse("not json", None, self.model)

    api.app.state.client_factory = lambda spec, settings: Garbage()  # type: ignore[attr-defined]
    assert post(api).status_code == 502
    assert api.app.state.breaker.failures == 0  # type: ignore[attr-defined]


def test_breaker_reopens_when_the_trial_call_fails() -> None:
    now = [0.0]
    b = CircuitBreaker(failures_to_open=1, cooldown_s=10, clock=lambda: now[0])
    b.failure()
    with pytest.raises(CircuitOpenError):
        b.before_call()
    now[0] = 10.0
    b.before_call()  # half-open: one trial allowed
    b.failure()
    assert b.state == "open" and b.opened_at == 10.0


def test_keys_and_ledger() -> None:
    keys = parse_keys(f" a={digest('k1')}:5 ,b={digest('k2').upper()}:7")
    assert [k.name for k in keys] == ["a", "b"] and keys[1].digest == digest("k2")
    assert authenticate("k2", keys) == keys[1]
    assert authenticate("nope", keys) is None and authenticate(None, keys) is None
    with pytest.raises(ValueError, match="name=sha256:budget"):
        parse_keys("broken")
    with pytest.raises(ValueError, match="64-character"):
        parse_keys("a=abc:5")
    day = [date(2026, 9, 29)]
    ledger = Ledger(today=lambda: day[0])
    key = ApiKey("a", digest("k1"), 100)
    ledger.charge(key, 80)
    assert ledger.remaining(key) == 20
    day[0] = date(2026, 9, 30)
    assert ledger.remaining(key) == 100


def test_keygen(capsys: pytest.CaptureFixture[str]) -> None:
    assert api_main(["keygen", "ci-bot", "--budget", "9000"]) == 0
    out = capsys.readouterr().out
    key = out.split("X-API-Key): ")[1].split()[0]
    assert key.startswith("gr_") and f"ci-bot={digest(key)}:9000" in out
    assert parse_keys(out.split("in .env:")[1].strip())[0].daily_tokens == 9000
    with pytest.raises(SystemExit):
        api_main(["keygen", "bad=name"])
