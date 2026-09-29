# ADR-016: A generation API with keys, budgets and a circuit breaker

- **Status:** accepted
- **Date:** 2026-09-29

## Context

Drafts were CLI-only. A service needs to expose generation without letting
one caller exhaust a provider quota, run up cost, or hammer a provider that
is already failing.

## Decision

`/v1` sits beside the evidence API (OpenAPI docs at `/docs`):

- `POST /v1/drafts` runs the **same pipeline as the CLI** (`generation.service`):
  retrieval, the prompt registry, redaction, the semantic cache and the full
  grounding gate. The response lists kept bullets with their entailment and
  dropped ones with their reason. `GET /v1/prompts`, `GET /v1/usage`,
  `GET /v1/health` (circuit state).
- **API keys**: `X-API-Key`; `.env` holds only SHA-256 digests
  (`API_KEYS=name=digest:daily_tokens`), compared in constant time.
  `python -m grounded.api keygen <name>` prints a key once.
- **Per-key daily token budgets**: a key over budget gets 429 before any
  model is called; tokens are charged from the provider's reported usage; a
  cache hit costs nothing.
- **Rate limiting** with slowapi, per key (`RATE_LIMIT`, default 10/minute).
- **Circuit breaker**: after `CIRCUIT_FAILURES` consecutive provider errors
  the circuit opens and requests get 503 with `Retry-After` without touching
  the provider; after `CIRCUIT_COOLDOWN_S` one trial call decides. Output that
  fails to parse does not count - the provider answered.

## Consequences

- The ledger, limiter and breaker are in memory, per process: correct for one
  instance, not for several. Step 60 (auth and multi-tenancy) moves budgets
  and keys into Postgres.
- The server still binds to loopback by default; the admin UI has no user
  authentication until step 60.
