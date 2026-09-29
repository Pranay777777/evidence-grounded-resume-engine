# ADR-015: Redact before any external call; a fully local mode

- **Status:** accepted
- **Date:** 2026-09-29

## Context

Free OpenRouter models may log prompts (ADR-007). Evidence can contain a
person's contact details, colleagues' names, or an employer's confidential
project names. Until now the only protection was "keep it out of evidence".

## Decision

- Before a request goes to **any provider that is not local**, the job
  description and every evidence body are redacted: each sensitive span
  becomes a placeholder (`[EMAIL_1]`, `[TERM_2]`), consistently within the
  request. Record IDs are never touched, so citations still resolve. The
  model's bullets are restored before the grounding gate, which compares them
  with the real evidence - redaction cannot weaken grounding.
- Engines (`REDACTION`): `patterns` (default, no dependencies - e-mail,
  phone, URL, IP, plus `REDACT_TERMS`, a deny-list for employer and client
  names) or `presidio` (extra `[privacy]` and `en_core_web_sm`; adds people's
  names). Presidio's ORGANIZATION and LOCATION entities are deliberately not
  used: spaCy tags technologies (Airflow, Snowflake) as organisations, which
  would hide the skills the bullets are about. Named terms cover employers.
- **Local mode**: `LLM_MODEL=ollama:<model>` sends nothing off the machine, so
  nothing is redacted; `LOCAL_ONLY=true` makes any non-local spec an error
  rather than a silent fallback.

## Consequences

- Placeholders cost the model some context ("[PERSON_1] mentored...").
- The small spaCy model misses some names (it did not tag an unusual first
  name in testing); `en_core_web_lg` is better and much larger. The deny-list
  is the dependable control for names you know.
- The red-team suite checks that contact details and deny-listed terms never
  reach the outbound prompt (LLM02).
