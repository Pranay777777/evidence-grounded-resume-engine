# ADR-020: A public demo that cannot be vandalised or bankrupted

- **Status:** accepted
- **Date:** 2026-10-01

## Context

Phase 2's exit criteria include a reachable live demo. A public page in
front of a model call has two failure modes: visitors changing what it
shows, and visitors spending the model quota until it stops working.

## Decision

- **Hugging Face Spaces** on the free tier. The Docker SDK and CPU Basic are
  paid for this account, so the Space uses the **Gradio SDK on ZeroGPU only as
  a free runtime** (no GPU is used; `app.py` registers one unused
  `@spaces.GPU` function because ZeroGPU stops Spaces without one):
  `deploy/huggingface/app.py` starts the project's own server on port 7860
  and no Gradio UI is used. `requirements.txt` installs a **tagged release**
  from GitHub and `app.py` fetches the demo data at the same tag, so the demo
  runs exactly the tagged code. The ONNX embedding and NLI models download on
  first start (not at build time), so a cold start takes a few minutes.
- **SQLite and the synthetic career**: `python -m grounded.demo` migrates,
  loads `benchmarks/retrieval/corpus.yaml` and indexes it on start. The same
  evidence the benchmarks use, so a reader can check the README's numbers
  against what the demo shows.
- **Read-only** (`DEMO=true`): visitors get `read_evidence` only - every
  write returns 403 - and the admin is not served.
- **Samples need no model.** The UI offers the 20 synthetic job
  descriptions; each replays a draft a real model wrote (from the golden
  set) through the live gate. The demo stays useful when the free quota is
  gone, and shows dropped bullets with their reasons.
- **Own job descriptions are bounded**: a per-visitor rate limit
  (`DEMO_RATE`, keyed by the platform proxy's `X-Forwarded-For`, trusted only
  in demo mode) and a daily token budget for the whole demo
  (`DEMO_DAILY_TOKENS`), checked before any model call. The provider key is a
  Space secret.

## Consequences

- SQLite in `/tmp` is rebuilt on every restart (models are re-downloaded too): nothing a visitor does
  persists, by design.
- The Gradio runtime installs its own pinned dependencies beside ours
  (Gradio 6.29's `mcp` extra caps pydantic at 2.12.5), so the package's
  dependency floors must stay compatible with the Space's Gradio version.
- Free models may refuse or rate-limit; the page says so and points to the
  samples.
