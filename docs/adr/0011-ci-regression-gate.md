# ADR-011: CI re-measures the gate on frozen data and fails on regression

- **Status:** accepted
- **Date:** 2026-09-29

## Context

The grounding gate's quality is a measured number (ADR-009): on 21 labelled
pairs, the full gate at 0.95 makes no false accepts and no false rejects. A
change to the claim-strength lexicon, the premise format, the threshold or
the NLI model could quietly break that, and the unit tests would not notice,
because they use a scripted verifier.

## Decision

A CI job, **Eval regression gate**, runs `make eval` on every push:

1. `python -m grounded.verification calibrate benchmarks/verifier/pairs.yaml
   --max-false-accept 0 --max-false-reject 0` - the real NLI model on the
   frozen pairs; exit 1 if the **full gate at the production threshold**
   breaks a limit.
2. `python -m grounded.evals run --check benchmarks/golden/limits.yaml
   --skip-if-unlabelled` - the golden set against its limits, once it has
   labels (ADR-010). Until then this step reports that it skipped.

Both reports go to the job summary.

- **Never calls an LLM.** Generation is not re-run in CI: it is slow, costs
  quota, and free models change or disappear (ADR-007). The gate is
  evaluated against frozen generations and frozen pairs, which makes the job
  deterministic.
- **The only download is the NLI model** (`Xenova/nli-deberta-v3-xsmall`,
  quantised ONNX), cached with `actions/cache`, keyed on `nli.py` so a model
  change refreshes it.
- **Limits are zero.** The set is small, so each pair is a meaningful share
  of a rate; any change is a real change, and a PR that moves a number must
  update the pairs or the limits and say why in an ADR. The margin is not
  thin: supported pairs score 0.98-0.99 against a 0.95 threshold, and no
  unsupported pair the checks miss scores above 0.56.

## Consequences

- The zero false-accept result partly rests on a lexicon written after
  seeing the failures (ADR-008 amendment). The gate protects that result from
  regressing; it does not make it an out-of-sample number. The golden set is
  what will.
- The job adds a few minutes to CI, mostly the first model download.
- The calibration file carries a layer-by-layer table at the production
  threshold, so each check's contribution is re-measured on every run and
  recorded in `docs/results/improvement-curve.md` (step 53).
