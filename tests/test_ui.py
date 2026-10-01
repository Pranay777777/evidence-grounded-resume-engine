"""The draft UI (step 62): inline citations, evidence on hover, base-vs-tailored diff."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from grounded.api.app import create_app
from grounded.config import get_settings
from grounded.evidence.models import Base
from grounded.evidence.store import load, read_file
from grounded.generation.diff import added_share, word_diff
from grounded.generation.llm import ChatResponse, LLMError
from grounded.retrieval.embedding import get_embedder
from grounded.retrieval.index import embed_pending
from grounded.verification.nli import Label, Verdict

FIXTURE = Path(__file__).parent / "fixtures" / "retrieval_corpus.yaml"
JD = "Data engineer to build incremental Delta Lake loads and HNSW search."


class Entailed:
    name = "fake-nli"

    def check(self, premise: str, hypothesis: str) -> Verdict:
        return Verdict(Label.ENTAILMENT, 0.96)


class TwoBullets:
    model = "fake/model:free"

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        bullets = [
            {
                "text": "Implemented incremental Delta Lake loads with MERGE, cutting reruns.",
                "evidence_ids": ["ex-delta-merge", "ex-hnsw-index"],
            },
            {"text": "Architected the <b>whole</b> platform.", "evidence_ids": ["ex-delta-merge"]},
        ]
        return ChatResponse(None, json.dumps({"bullets": bullets}), self.model)


@pytest.fixture
def ui(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    monkeypatch.setenv("EMBEDDER", "hashing")
    get_settings.cache_clear()
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        load(s, read_file(FIXTURE))
        embed_pending(s, get_embedder("hashing"))
        s.commit()
    app = create_app(
        engine, client_factory=lambda spec, settings: TwoBullets(), verifier=lambda: Entailed()
    )
    with TestClient(app) as client:
        yield client
    get_settings.cache_clear()


def token(client: TestClient) -> str:
    page = client.get("/ui").text
    match = re.search(r'name="csrf_token" value="([^"]+)"', page)
    assert match is not None
    return match.group(1)


def test_the_form_renders(ui: TestClient) -> None:
    page = ui.get("/ui")
    assert page.status_code == 200 and 'action="/ui/draft"' in page.text


def test_a_draft_shows_citations_evidence_and_the_diff(ui: TestClient) -> None:
    page = ui.post("/ui/draft", data={"job_description": JD, "csrf_token": token(ui)})
    assert page.status_code == 200, page.text
    html = page.text
    assert "Implemented incremental Delta Lake loads with MERGE, cutting reruns." in html
    assert '>[1]<span class="pop"' in html and '>[2]<span class="pop"' in html  # numbered, inline
    assert "ex-delta-merge</strong> · rev 1" in html  # the hover card names the record
    assert "<ins>cutting reruns.</ins>" in html  # what the model added, highlighted
    assert "Base (evidence):" in html and "of words not in the evidence" in html
    assert "Dropped (1)" in html and "bullets are plain text" in html
    assert "&lt;b&gt;whole&lt;/b&gt;" in html and "<b>whole</b>" not in html  # autoescaped


def test_csrf_and_input_errors(ui: TestClient) -> None:
    assert ui.post("/ui/draft", data={"job_description": JD}).status_code == 403
    short = ui.post("/ui/draft", data={"job_description": "too short", "csrf_token": token(ui)})
    assert short.status_code == 422 and "at least 20 characters" in short.text


def test_model_failures_and_missing_evidence_are_shown(ui: TestClient) -> None:
    def broken(spec: str, settings: Any) -> Any:
        raise LLMError("provider returned 503: down")

    ui.app.state.client_factory = broken  # type: ignore[attr-defined]
    failed = ui.post("/ui/draft", data={"job_description": JD, "csrf_token": token(ui)})
    assert failed.status_code == 502 and "provider returned 503" in failed.text


def test_no_evidence_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDER", "hashing")
    get_settings.cache_clear()
    empty = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(empty)
    app = create_app(
        empty, client_factory=lambda spec, settings: TwoBullets(), verifier=lambda: Entailed()
    )
    with TestClient(app) as client:
        page = client.post("/ui/draft", data={"job_description": JD, "csrf_token": token(client)})
        assert page.status_code == 422 and "no verified evidence" in page.text
    get_settings.cache_clear()


def test_word_diff() -> None:
    diff = word_diff(
        "Wrote Airflow DAGs from twelve source systems.",
        "Built Airflow DAGs from twelve systems, cutting failures.",
    )
    assert diff == [
        ("removed", "Wrote"),
        ("added", "Built"),
        ("same", "Airflow DAGs from twelve"),
        ("removed", "source"),
        ("same", "systems,"),
        ("added", "cutting failures."),
    ]
    assert added_share(diff) == pytest.approx(3 / 8)
    assert word_diff("", "") == [] and added_share([]) == 0.0
