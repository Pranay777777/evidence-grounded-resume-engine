# ADR-009: An NLI entailment verifier decides which bullets survive

- **Status:** accepted, amended 2026-09-28 (threshold, premise)
- **Date:** 2026-09-28

## Context

The first live draft cited three real, verified records. Every bullet still
embellished: "CI" became "CI/CD with GitHub Actions"; "the pipeline
processes" became "developed a pipeline"; masking "in the Silver layer"
gained "using Python on Delta Lake". None of these contradicts its evidence.
Each is simply *not supported* by it — and no deterministic check can see
that. This is the problem the project exists to solve.

## Decision

### A natural-language-inference model checks every bullet

The cited statements are the premise; the bullet is the hypothesis. The model
classifies the pair as entailment, neutral or contradiction. A bullet
survives only if the label is **entailment** and its probability meets the
threshold. **Neutral fails** — that is the embellishment case — as does
contradiction.

### The model: `Xenova/nli-deberta-v3-xsmall`

DeBERTa-v3, fine-tuned for NLI on SNLI and MultiNLI, exported to ONNX and run
with onnxruntime and a Hugging Face tokenizer — CPU-only, no PyTorch, in the
same `[embeddings]` extra as retrieval. The label order is read from the
checkpoint's own config, never assumed: NLI checkpoints disagree about it,
and a swapped index would silently turn the verifier into its opposite.

### The premise is the cited statements only

A record's project summary and skills tags help retrieval find it, but they
are not verified facts. "Delta Lake" in a project summary cannot make
"…using Delta Lake" true. A test pins this.

### The threshold prefers false negatives, and is measured

A wrongly rejected true bullet costs a line; a wrongly accepted false one
costs the candidate's credibility in an interview. The default is 0.8, and it
is not a guess to leave alone: `python -m grounded.verification calibrate`
runs 21 labelled pairs — including the three real embellished bullets,
verbatim — across a sweep of thresholds and reports the false-accept and
false-reject rates (docs/results/verifier-calibration.md). The threshold is
set from that table.

### Why not an LLM judge

A judge model reading premise and bullet is the obvious alternative, and may
be stronger on subtle cases. It was not chosen as the gate because the free
provider routes each request to a random model — a verifier that changes
identity per call cannot be calibrated — and because a language model
grading another's output shares its failure modes, including the
temptation to accept fluent overstatement. It remains a candidate for a
second opinion once a pinned model is available (step 55).

## Consequences

**Gained:** the embellishments that make grounded generation hard — scope
inflation, borrowed technologies, invented outcomes — are checked by a model
whose error rates are measured and published, not asserted.

**Given up:** a 22M-parameter NLI model is not a reasoner: it can miss a
subtle overstatement or reject a faithful paraphrase, and the calibration
set is small (21 pairs) until the golden set (step 50) replaces it. Every
bullet costs one inference. The model was not runnable in the build
environment (no access to its host), so its real numbers come from the
developer's machine — the adapter is tested against a fake session.

## Amendment — threshold 0.95, and project names in the premise

**Threshold.** The first calibration (21 pairs, `nli-deberta`) gave, NLI
alone: false accept 31% at 0.8 and 15% at 0.95, with **no supported pair
rejected at any threshold**. Raising the bar cost nothing measurable, so the
default is now 0.95. The residual 15% is the scope-inflation blind spot,
which ADR-008's claim-strength check now covers.

**Premise.** The first verified live draft dropped a faithful bullet —
"…masking into the lakehouse's Silver layer *for the metadata-driven-lakehouse
project*" — at entailment 0.00, because the premise never named the project.
A record's project link is part of its verified fact (it is in the content
hash), so the premise now appends "This was part of the <project> project."
for each cited record's project. Summaries and skills tags stay out: they
are not hashed, so not verified.
