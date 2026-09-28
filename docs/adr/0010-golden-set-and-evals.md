# ADR-010: A human-labelled golden set measures the gate, not the model

- **Status:** accepted
- **Date:** 2026-09-28

## Context

Calibration (ADR-009) measured the verifier on 21 hand-written pairs. That
says how the gate behaves on sentences written to test it — not on what a
model actually writes for a job description. The headline claim of this
project, "unsupported claims do not reach the résumé", needs a number
measured on real generations, judged by a person.

## Decision

### The golden set is raw generations plus one human judgement each

`python -m grounded.evals collect --model <pin>` runs every synthetic job
description in `benchmarks/golden/jds.yaml` through retrieval and generation
against the synthetic career (`benchmarks/retrieval/corpus.yaml`) and stores
**every** bullet, ungated, in `benchmarks/golden/golden.jsonl`. Collection is
deliberately ungated: the set exists to measure the gate, so it must contain
what the gate is meant to catch.

Each item stores the job, model, prompt version, bullet, citations and the
**premise exactly as the verifier sees it** (cited statements + project
names, ADR-009), so it can be judged and re-scored later without a database
or an LLM call.

`python -m grounded.evals label` asks one question per bullet — *does the
premise, on its own, support every claim?* — and saves after each answer.
The judgement is about words, not spirit: "CI/CD" is not supported by "CI".

### Rules that keep the set trustworthy

- **A pinned model is required.** `openrouter/free` picks a model per
  request; a set from an unknown mixture of models cannot be regenerated or
  compared (ADR-007). Each item records its model.
- **Item IDs are content-addressed** (`<jd>-<hash(model, text)>`), and
  re-collecting only *appends*. Positional IDs would let a second run
  overwrite or hide a labelled bullet; with content IDs, a labelled bullet is
  never lost and the same bullet is never labelled twice.
- **Collection is incremental.** Free models are rate limited upstream (the
  first real run got a 429 on its first call). Each job's bullets are saved as
  they arrive; a provider error skips that job; three in a row stop the run
  without spending more calls; and a rerun skips jobs the model already
  answered, so quota is spent only on what is missing.
- **Synthetic data only.** The career and the job descriptions are fictional:
  free OpenRouter models may log prompts, and real evidence is not loaded yet.
  Several job descriptions ask for things the career lacks (Kubernetes, line
  management, an AWS certification) — a grounded generator must leave them out.

### Metrics (`python -m grounded.evals run`)

The stored bullets go back through the production gate — citation resolves →
numbers → claim strength → NLI ≥ threshold — against the stored premise.
Citation currency is not re-checked: the premise is frozen at collection.

| Metric | Definition |
|---|---|
| **Output fabrication rate** (headline) | unsupported ÷ kept — what a reader would actually see; its complement is faithfulness |
| Raw fabrication rate | unsupported ÷ labelled — what the model writes |
| Gate false accept / false reject | on real generations, complementing ADR-009's pairs |
| Citation precision / recall | kept bullets' citations against each JD's labelled `relevant` records |
| JD keyword coverage | share of *attainable* JD terms (ones some evidence uses) that kept bullets use; unattainable terms are excluded because omitting them is correct |
| Tone consistency | rule-based (no first person, capitalised verb-style opener, one sentence); crude and labelled as such |

The report lists the IDs of **false accepts** first: they are the failures
that matter.

### Regression limits

`benchmarks/golden/limits.yaml` holds `max:`/`min:` limits on report fields;
`run --check` exits 1 on a breach and 2 on an unknown metric name. The
shipped limits are **targets**, not measurements (output fabrication ≤ 5%,
false accept ≤ 10%); they are re-set from the first labelled run. In CI
(step 52) the check runs on the frozen, labelled set — deterministic, and
never calling an LLM.

## Consequences

- The fabrication number rests on one labeller's judgement. Labels record who
  and when; a second labeller and agreement rate would strengthen it.
- 20 job descriptions × one model may give fewer than 100 bullets; a second
  pinned model or more job descriptions extends the set without touching
  existing labels.
- Changing the premise format (ADR-009) invalidates stored premises; the set
  would be re-collected, not edited.
