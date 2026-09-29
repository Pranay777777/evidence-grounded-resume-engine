# ADR-014: A red-team suite that assumes the model is fooled

- **Status:** accepted
- **Date:** 2026-09-29

## Context

Job descriptions are untrusted input: whoever wrote the posting controls text
that reaches the model. The OWASP Top 10 for LLM Applications 2026 (published
4 August 2026) keeps Prompt Injection at LLM01 and is explicit that no
prompt-level defence is reliable - the goal is a system in which a fooled
model cannot do damage.

## Decision

`benchmarks/redteam/payloads.yaml` holds 34 hostile job descriptions mapped to
OWASP 2026 ids (LLM01, 02, 06, 07, 08, 09, 10). `python -m grounded.evals
redteam` runs them in two modes.

**Offline, worst case (CI).** For each `gate` payload the suite writes the
bullet a *fully compromised* model would produce - the injected claim, citing
a real or forged record - and runs it through the production gate. It passes
only if the gate drops it. This measures the property that matters (what
reaches the reader) without depending on whether today's model happens to
resist the injection, and it needs no LLM call. Prompt checks (`fence`,
`invisible`, `length`, `redaction`) run against every registered prompt.
`make eval` runs it with `--min-pass 1.0`.

**Live.** `--live --model <pin>` sends the `gate` payloads to a real model and
reports *obeyed* (the model wrote the injected claim) and *reached output* (a
kept bullet carries it) separately. The first is the model's weakness; the
second is the system's.

### Controls the suite led to

- **`generate-v2`** (ADR-012): untrusted text can no longer close or open a
  prompt fence (`</job_description>` is neutralised) and invisible characters
  (zero-width, the Unicode tag block, variation selectors) are stripped. The
  suite shows `generate-v1` failing both; v2 becomes the default. v1 stays
  registered, so earlier results remain reproducible.
- **A markup check in the gate** (LLM10): an entailment model judges meaning,
  not format, so `<script>...</script> Wrote Airflow DAGs.` can be entailed.
  HTML tags, Markdown links and images, code spans, control characters and
  URLs not present in the evidence now drop a bullet before NLI runs. None of
  the 213 golden-set bullets trips it, so earlier results are unchanged.
- **An output-token cap** on every call (`LLM_MAX_TOKENS`, LLM06).

Not exercised: LLM03 (the model's only tool returns data; nothing to misuse),
LLM04 and LLM05 (covered by pip-audit and gitleaks, and by human verification
of evidence, ADR-002).

## Consequences

- With a verifier that accepts everything, the deterministic checks alone
  still stop 19 of the 29 gate payloads; the other 10 rest on the NLI model.
  Those are the cases to watch when the verifier changes.
- The worst-case attacks are written by the author. They are a floor, not a
  proof: a novel paraphrase that NLI entails would pass. The live mode and a
  growing payload file are how that floor rises.
