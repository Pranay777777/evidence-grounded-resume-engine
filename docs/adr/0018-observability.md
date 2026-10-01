# ADR-018: OpenTelemetry traces with the gate's verdict as the eval signal

- **Status:** accepted
- **Date:** 2026-10-01

## Context

A draft passes through retrieval, an optional cache, a model call with
retries, and the gate. When one is slow, expensive or wrong, the question is
always the same: which request, which model and prompt, which stage, how
many tokens - and did the output hold up?

## Decision

- **OpenTelemetry**, not a vendor SDK: one span tree per request, exportable
  to any OTLP backend - Jaeger, Tempo, Honeycomb, or Langfuse, which ingests
  OTLP. `OTEL_EXPORTER=none|console|otlp`; with `none` (the default) the API's
  no-op tracer makes instrumentation free. The SDK and OTLP exporter are the
  `observability` extra.
- Spans: `HTTP <method> <route>` > `draft` > `retrieve`, `cache.lookup`,
  `llm.generate`, `gate`. Each span's duration is that stage's latency.
- Attributes: request ID (`X-Request-ID` honoured if safe, else generated, and
  echoed), tenant, model requested and served, prompt version and fingerprint
  (ADR-012), **retrieval version** (embedder, fusion, reranker, k), attempts,
  retries, tool calls, input and output tokens, **cost** ($0 on free tiers,
  from `LLM_*_PRICE_PER_MTOK` otherwise, absent if unknown), redacted spans.
  GenAI semantic-convention names (`gen_ai.*`) where they exist.
- **Eval signal**: what the gate decided - kept, dropped, drop reasons as
  fixed categories, the share kept, mean entailment. It is the online twin
  of the offline evals and needs no labels.
- One JSON log line per request with the same request ID, so logs and
  traces join.

## Consequences

- No bullet text, job description or evidence goes into spans or logs - only
  counts, IDs and categories - so traces are safe to ship to a third party.
- Cost for paid models is only as right as the configured prices.
