# ADR-013: A semantic cache that cannot weaken grounding

- **Status:** accepted
- **Date:** 2026-09-29

## Context

The same request often arrives twice - a job description re-pasted,
reformatted or lightly edited. Each draft costs a model call: quota on free
tiers, money on paid ones, and seconds of latency. A cache keyed on exact
text misses every reformatting; a cache keyed on similarity alone could hand
back a draft written for a different job.

## Decision

A draft is reused only when **all** of these match an earlier request:

1. the model spec and the prompt fingerprint (ADR-012);
2. the **exact evidence set** - record IDs and revisions. Citations only
   mean something against the records a draft was written from, and an
   edited record is a new revision (ADR-002), so an edit is always a miss;
3. the job description: identical after whitespace and case folding, or
   embedding cosine similarity >= `CACHE_THRESHOLD`.

A cached draft goes through the **full grounding gate on every use**. A hit
saves a model call; it can never let an unverified bullet through.

The cache is opt-in (`draft --cache`), stored in `.cache/semantic-cache.jsonl`
(gitignored). `evals collect` never uses it: the golden set must be fresh
generations. `python -m grounded.generation cache-stats` reports hits and
tokens saved.

### Measured, not assumed

`python -m grounded.generation cache-bench benchmarks/cache/pairs.yaml`
scores 24 labelled pairs (12 the same request reformatted or paraphrased,
12 different requests) with the configured embedder and no model calls. For
each threshold it reports the hit rate on the same-request pairs and the
false-hit rate on the rest - on similarity alone and with the evidence rule -
and projects tokens and dollars saved from the tokens drafts really used
(runs.jsonl). The threshold default (0.95) is provisional until that run;
the result is `docs/results/semantic-cache.md`.

## Amendment - threshold from the first benchmark run (2026-09-29)

`cache-bench` with `bge-small` on the 24 pairs:

| Threshold | Hit rate (similarity) | False hits (similarity) | Hit rate (+ same evidence) | False hits (+ same evidence) |
|---|---|---|---|---|
| 0.80 | 100% | 25% | 50% | 0% |
| 0.85 | 100% | 17% | 50% | 0% |
| **0.90** | 75% | **0%** | **50%** | **0%** |
| 0.95 | 50% | 0% | 42% | 0% |
| 0.97 | 33% | 0% | 33% | 0% |

- **The evidence rule is what keeps hits correct**: with it, no threshold
  produced a false hit. Similarity alone reaches 0% false hits at 0.90.
- The default moves from the provisional 0.95 to **0.90** - the lowest
  threshold where either safeguard alone would have prevented every false
  hit - raising the hit rate on repeated requests from 42% to 50%.
- The evidence rule costs hits: half the paraphrases retrieve a different
  record set and miss. That is the intended trade (see Decision).
- Caveat: 24 pairs, written by the author and evaluated in-sample. Savings
  projected from real drafts: ~1,150 tokens per draft; $0 on free models.

## Consequences

- The evidence rule is conservative: a paraphrase that retrieves even one
  different record misses. The benchmark shows how much hit rate that costs;
  it is the price of never serving citations for records the model did not
  see.
- A file-backed cache fits the CLI; the service (step 59) can move it to
  Postgres + pgvector behind the same interface.
