"""Traces for every draft (ADR-018).

OpenTelemetry, so any backend that speaks OTLP can receive them - Jaeger,
Grafana Tempo, Honeycomb, or Langfuse (which ingests OTLP). One trace per
request:

    HTTP POST /v1/drafts            request ID, route, status, tenant
      draft                         model, prompt version + fingerprint, retrieval version,
                                    tokens, cost, cache hit, eval signal
        retrieve                    k, rerank, hits
        cache.lookup                hit, similarity
        llm.generate                gen_ai.* attributes, attempts, retries, tool calls,
                                    tokens, redacted spans
        gate                        kept, dropped, drop reasons, mean entailment

Each span's duration is that stage's latency. The **eval signal** is what
the gate decided: the share of bullets kept and their mean entailment - a
live proxy for draft quality, the production twin of the offline evals.

Attribute names follow OpenTelemetry's GenAI semantic conventions where one
exists (`gen_ai.request.model`, `gen_ai.usage.input_tokens`) and use a
`grounded.` prefix otherwise. With no exporter configured (`OTEL_EXPORTER=none`,
the default) the API's no-op tracer makes all of this free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opentelemetry import trace

from grounded import __version__

if TYPE_CHECKING:
    from grounded.config import Settings

tracer = trace.get_tracer("grounded", __version__)


def cost_usd(model: str, usage: dict[str, int], settings: Settings) -> float | None:
    """$0 on free tiers; from configured prices otherwise; unknown if unpriced."""
    if model.endswith(":free"):
        return 0.0
    if settings.llm_input_price_per_mtok is None or settings.llm_output_price_per_mtok is None:
        return None
    return (
        usage.get("prompt_tokens", 0) * settings.llm_input_price_per_mtok
        + usage.get("completion_tokens", 0) * settings.llm_output_price_per_mtok
    ) / 1_000_000


def retrieval_version(settings: Settings, k: int, rerank: bool) -> str:
    """Everything that decides which evidence a prompt sees, in one string."""
    return f"{settings.embedder}+bm25+rrf60{'+minilm-rerank' if rerank else ''}@k{k}"


def configure_tracing(settings: Settings, install: bool = True) -> Any:
    """Install an SDK tracer provider for the configured exporter.

    Needs the `observability` extra (the SDK); without an exporter, nothing is
    installed and the no-op tracer stays in place.
    """
    if settings.otel_exporter == "none":
        return None
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError as exc:  # pragma: no cover - the SDK ships with the dev extra
        raise ImportError('OTEL_EXPORTER needs: pip install -e ".[observability]"') from exc

    provider = TracerProvider(
        resource=Resource.create({"service.name": "grounded", "service.version": __version__})
    )
    if settings.otel_exporter == "console":
        exporter: Any = ConsoleSpanExporter()
    else:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError as exc:
            raise ImportError(
                'OTEL_EXPORTER=otlp needs: pip install -e ".[observability]"'
            ) from exc
        exporter = OTLPSpanExporter()  # endpoint from OTEL_EXPORTER_OTLP_ENDPOINT
    provider.add_span_processor(BatchSpanProcessor(exporter))
    if install:
        trace.set_tracer_provider(provider)
    return provider
