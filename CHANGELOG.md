# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.2] - 2026-10-01

### Fixed
- The public demo's forms inside the Hugging Face page: the CSRF cookie was
  `SameSite=Strict`, which a cross-site iframe never sends. In demo mode it is
  now `SameSite=None; Secure; Partitioned`, and a browser that blocks it falls
  back to the Origin check (ADR-020).

## [1.0.1] - 2026-10-01

### Fixed
- The Hugging Face Space build: the pydantic floor is now 2.11.10 (was 2.13.5),
  so the package installs beside Gradio 6.29, which caps pydantic at 2.12.5.
  Tests pass on pydantic 2.12.5; no code changed (ADR-020).

## [1.0.0] - 2026-10-01

First release - see [docs/releases/v1.0.0.md](docs/releases/v1.0.0.md).

### Added
- Evidence store with atomic, versioned, human-verified records (ADR-002, ADR-003).
- Hybrid retrieval with reranking and a published ablation (ADR-004 to ADR-006).
- Structured generation through OpenRouter, Ollama or OpenAI behind one adapter spec,
  with a versioned prompt registry (ADR-007, ADR-012).
- The grounding gate: citation, markup, number, claim-strength and NLI checks (ADR-008, ADR-009, ADR-014).
- Golden set, eval harness, CI regression gate and improvement curve (ADR-010, ADR-011).
- Semantic cache that never weakens grounding (ADR-013).
- OWASP LLM Top 10 2026 red-team suite; redaction and local-only mode (ADR-014, ADR-015).
- Generation API with API keys, budgets, rate limits and a circuit breaker (ADR-016).
- JWT scopes and tenant isolation in the ORM and Postgres RLS (ADR-017).
- OpenTelemetry tracing with the gate's verdict as an eval signal (ADR-018).
- Draft UI with inline citations, evidence on hover and a base-vs-tailored diff (ADR-019).
- Public read-only demo on Hugging Face Spaces (ADR-020).

[Unreleased]: https://github.com/Pranay777777/evidence-grounded-resume-engine/compare/v1.0.2...HEAD
[1.0.2]: https://github.com/Pranay777777/evidence-grounded-resume-engine/releases/tag/v1.0.2
[1.0.1]: https://github.com/Pranay777777/evidence-grounded-resume-engine/releases/tag/v1.0.1
[1.0.0]: https://github.com/Pranay777777/evidence-grounded-resume-engine/releases/tag/v1.0.0
