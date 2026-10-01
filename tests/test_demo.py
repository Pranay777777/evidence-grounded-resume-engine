"""The public demo (step 63, ADR-020): read-only, rate limited, samples need no model."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from grounded import demo
from grounded.api import drafts
from grounded.api.app import create_app
from grounded.api.ui import samples
from grounded.config import Settings, get_settings
from grounded.generation.llm import ChatResponse
from grounded.verification.nli import Label, Verdict


class Entailed:
    name = "fake-nli"

    def check(self, premise: str, hypothesis: str) -> Verdict:
        return Verdict(Label.ENTAILMENT, 0.97)


class Counting:
    calls = 0
    model = "fake/model:free"

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        Counting.calls += 1
        bullets = [
            {
                "text": "Wrote Airflow DAGs orchestrating nightly loads.",
                "evidence_ids": ["b-airflow-dags"],
            }
        ]
        return ChatResponse(
            None,
            json.dumps({"bullets": bullets}),
            self.model,
            usage={"prompt_tokens": 900, "completion_tokens": 100},
        )


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    url = f"sqlite:///{tmp_path / 'demo.db'}"
    for key, value in {
        "DEMO": "true",
        "EMBEDDER": "hashing",
        "DATABASE_URL": url,
        "DEMO_RATE": "100/hour",
        "DEMO_DAILY_TOKENS": "1500",
        "SEMANTIC_CACHE_PATH": str(tmp_path / "c.jsonl"),
    }.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    drafts.limiter.reset()
    assert demo.prepare(get_settings()) == 25
    Counting.calls = 0
    app = create_app(
        create_engine(url), client_factory=lambda spec, s: Counting(), verifier=lambda: Entailed()
    )
    with TestClient(app) as client:
        yield client
    get_settings.cache_clear()


def csrf(client: TestClient) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get("/ui").text)
    assert match is not None
    return match.group(1)


JD = "Data engineer to orchestrate Airflow batch pipelines."


def test_prepare_is_idempotent(site: TestClient) -> None:
    assert demo.prepare(get_settings()) == 25


def test_the_demo_is_read_only(site: TestClient) -> None:
    assert site.get("/evidence").status_code == 200
    put = site.put(
        "/evidence/x", json={"id": "x", "kind": "achievement", "statement": "Planted by a visitor."}
    )
    assert put.status_code == 403 and put.json()["detail"] == "the public demo is read-only"
    assert site.get("/admin").status_code == 404
    page = site.get("/ui").text
    assert "Public demo." in page and "100/hour per visitor" in page


def test_samples_replay_recorded_drafts_without_a_model_call(site: TestClient) -> None:
    gallery = samples()
    assert len(gallery) == 20 and all(s.items for s in gallery)
    first = gallery[0]
    page = site.post("/ui/sample", data={"jd_id": first.job.id, "csrf_token": csrf(site)})
    assert page.status_code == 200, page.text
    assert f'recorded draft for "{first.job.title}"' in page.text
    assert first.model in page.text and "Kept (" in page.text
    assert Counting.calls == 0
    missing = site.post("/ui/sample", data={"jd_id": "nope", "csrf_token": csrf(site)})
    assert missing.status_code == 404


def test_drafts_spend_the_daily_budget_then_stop(site: TestClient) -> None:
    assert (
        site.post("/ui/draft", data={"job_description": JD, "csrf_token": csrf(site)}).status_code
        == 200
    )
    assert (
        site.post("/ui/draft", data={"job_description": JD, "csrf_token": csrf(site)}).status_code
        == 200
    )
    spent = site.post("/ui/draft", data={"job_description": JD, "csrf_token": csrf(site)})
    assert spent.status_code == 429 and "model budget for the demo is used up" in spent.text
    assert Counting.calls == 2


def test_visitors_are_rate_limited_by_forwarded_address(
    site: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEMO_RATE", "1/hour")
    monkeypatch.setenv("DEMO_DAILY_TOKENS", "100000")
    get_settings.cache_clear()
    token = csrf(site)

    def draft(ip: str) -> int:
        return site.post(
            "/ui/draft",
            data={"job_description": JD, "csrf_token": token},
            headers={"X-Forwarded-For": f"{ip}, 10.0.0.1"},
        ).status_code

    assert [draft("1.1.1.1"), draft("1.1.1.1"), draft("2.2.2.2")] == [200, 429, 200]


def test_main(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.delenv("DEMO", raising=False)
    with pytest.raises(SystemExit):
        demo.main([])
    warmed: list[str] = []

    class Embedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            warmed.append("embedder")
            return [[0.0]]

    class Verifier:
        def check(self, premise: str, hypothesis: str) -> Verdict:
            warmed.append("verifier")
            return Verdict(Label.ENTAILMENT, 1.0)

    monkeypatch.setattr(demo, "get_embedder", lambda name: Embedder())
    monkeypatch.setattr(demo, "get_verifier", lambda name, cache_dir=None: Verifier())
    assert demo.main(["--warm"]) == 0 and warmed == ["embedder", "verifier"]

    served: list[Any] = []
    monkeypatch.setenv("DEMO", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(demo, "prepare", lambda settings: 25)
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: served.append(kw))
    assert demo.main([]) == 0 and served[0]["port"] == Settings().port
    get_settings.cache_clear()
