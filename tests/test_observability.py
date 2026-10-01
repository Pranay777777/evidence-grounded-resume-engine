"""Traces and request logs (step 61) - spans recorded in memory, no backend."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from grounded.api import drafts
from grounded.api.app import create_app
from grounded.api.guards import digest
from grounded.config import Settings, get_settings
from grounded.evidence.models import Base
from grounded.evidence.store import load, read_file
from grounded.generation.llm import ChatResponse
from grounded.logging import JsonFormatter
from grounded.observability import configure_tracing, cost_usd, retrieval_version
from grounded.retrieval.embedding import get_embedder
from grounded.retrieval.index import embed_pending
from grounded.verification.nli import Label, Verdict

FIXTURE = Path(__file__).parent / "fixtures" / "retrieval_corpus.yaml"
EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
trace.set_tracer_provider(_provider)


class Entailed:
    name = "fake-nli"

    def check(self, premise: str, hypothesis: str) -> Verdict:
        return Verdict(Label.ENTAILMENT, 0.97)


class Retrying:
    """Invalid output first, then a draft with one supported and one inflated bullet."""

    model = "fake/model:free"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        self.calls += 1
        if self.calls == 1:
            return ChatResponse("not json", None, self.model, usage={"prompt_tokens": 300})
        bullets = [
            {
                "text": "Implemented incremental loads with Delta Lake MERGE.",
                "evidence_ids": ["ex-delta-merge"],
            },
            {
                "text": "Led incremental loads with Delta Lake MERGE.",
                "evidence_ids": ["ex-delta-merge"],
            },
        ]
        return ChatResponse(
            None,
            json.dumps({"bullets": bullets}),
            self.model,
            usage={"prompt_tokens": 320, "completion_tokens": 80},
        )


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("EMBEDDER", "hashing")
    monkeypatch.setenv("API_KEYS", f"tracer={digest('gr_t')}:100000")
    monkeypatch.setenv("SEMANTIC_CACHE_PATH", str(tmp_path / "cache.jsonl"))
    get_settings.cache_clear()
    drafts.limiter.reset()
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        load(s, read_file(FIXTURE))
        embed_pending(s, get_embedder("hashing"))
        s.commit()
    app = create_app(
        engine, client_factory=lambda spec, settings: Retrying(), verifier=lambda: Entailed()
    )
    EXPORTER.clear()
    with TestClient(app) as client:
        yield client
    get_settings.cache_clear()


def ancestors(span: ReadableSpan) -> set[int]:
    by_id = {s.context.span_id: s for s in EXPORTER.get_finished_spans() if s.context}
    found: set[int] = set()
    parent = span.parent
    while parent is not None:
        found.add(parent.span_id)
        parent = by_id[parent.span_id].parent if parent.span_id in by_id else None
    return found


def spans() -> dict[str, ReadableSpan]:
    return {s.name: s for s in EXPORTER.get_finished_spans()}


def test_one_trace_per_draft_with_every_stage(
    api: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="grounded.request")
    response = api.post(
        "/v1/drafts",
        json={"job_description": "Data engineer with Delta Lake loads."},
        headers={"X-API-Key": "gr_t", "X-Request-ID": "req-123"},
    )
    assert response.status_code == 200 and response.headers["x-request-id"] == "req-123"

    found = spans()
    assert {"HTTP POST /v1/drafts", "draft", "retrieve", "llm.generate", "gate"} <= set(found)
    root, draft = found["HTTP POST /v1/drafts"], found["draft"]
    trace_ids = {s.context.trace_id for s in found.values()}
    assert len(trace_ids) == 1  # one trace for the whole request
    for child in ("retrieve", "llm.generate", "gate"):
        parent = found[child].parent
        assert parent is not None and draft.context is not None
        assert parent.span_id == draft.context.span_id
    # Newer FastAPI adds its own spans in between; what matters is ancestry.
    assert root.context is not None and root.context.span_id in ancestors(draft)

    assert root.attributes is not None and root.attributes["grounded.request_id"] == "req-123"
    assert root.attributes["http.response.status_code"] == 200
    a = draft.attributes
    assert a is not None
    assert a["grounded.prompt.version"] == "generate-v2"
    assert a["grounded.retrieval.version"] == "hashing+bm25+rrf60@k12"
    assert a["gen_ai.response.model"] == "fake/model:free"
    assert (a["gen_ai.usage.input_tokens"], a["gen_ai.usage.output_tokens"]) == (620, 80)
    assert (a["grounded.attempts"], a["grounded.retries"], a["grounded.tool_calls"]) == (2, 1, 2)
    assert a["grounded.cost_usd"] == 0.0
    assert (a["grounded.gate.kept"], a["grounded.gate.dropped"]) == (1, 1)
    assert a["grounded.eval.kept_ratio"] == 0.5 and a["grounded.eval.mean_entailment"] == 0.97
    assert a["grounded.gate.drop_reasons"] == ("claim_strength=1",)
    assert (
        found["retrieve"].attributes is not None
        and found["retrieve"].attributes["grounded.hits"] > 0
    )

    line = next(r for r in caplog.records if r.name == "grounded.request")
    fields = line.fields  # type: ignore[attr-defined]
    assert fields["request_id"] == "req-123" and fields["route"] == "/v1/drafts"
    assert fields["tokens"] == 700 and fields["kept"] == 1 and fields["tenant"] == "default"
    payload = json.loads(JsonFormatter().format(line))
    assert payload["msg"] == "request" and payload["status"] == 200


def test_unsafe_request_ids_are_replaced(api: TestClient) -> None:
    response = api.get("/v1/health", headers={"X-Request-ID": "bad id\nwith newline"})
    assert len(response.headers["x-request-id"]) == 32
    assert api.get("/v1/health").headers["x-request-id"] != response.headers["x-request-id"]


def test_a_cache_hit_is_traced_and_costs_nothing(api: TestClient) -> None:
    body = {"job_description": "Data engineer with Delta Lake loads.", "cache": True}
    api.post("/v1/drafts", json=body, headers={"X-API-Key": "gr_t"})
    EXPORTER.clear()
    api.post("/v1/drafts", json=body, headers={"X-API-Key": "gr_t"})
    found = spans()
    assert "llm.generate" not in found
    assert found["cache.lookup"].attributes is not None
    assert found["cache.lookup"].attributes["grounded.cache.hit"] is True
    assert (
        found["draft"].attributes is not None
        and found["draft"].attributes["grounded.cost_usd"] == 0.0
    )


def test_cost_and_retrieval_version() -> None:
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}
    assert cost_usd("m:free", usage, Settings()) == 0.0
    assert cost_usd("paid/m", usage, Settings()) is None
    priced = Settings(llm_input_price_per_mtok=0.5, llm_output_price_per_mtok=2.0)
    assert cost_usd("paid/m", usage, priced) == pytest.approx(1.5)
    assert (
        retrieval_version(Settings(embedder="bge-small"), 8, True)
        == "bge-small+bm25+rrf60+minilm-rerank@k8"
    )


def test_configure_tracing() -> None:
    assert configure_tracing(Settings(otel_exporter="none")) is None
    provider = configure_tracing(Settings(otel_exporter="console"), install=False)
    assert isinstance(provider, TracerProvider)
    try:
        import opentelemetry.exporter.otlp.proto.http  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"\[observability\]"):
            configure_tracing(Settings(otel_exporter="otlp"), install=False)
    else:
        assert configure_tracing(Settings(otel_exporter="otlp"), install=False) is not None
