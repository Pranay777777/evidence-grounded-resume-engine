# Improvement curve

Each change that moved a measured number, with the number before and after
on the **same** benchmark. Every figure comes from a committed result file;
nothing here is estimated.

## Retrieval - synthetic benchmark, 20 labelled queries

Source: [`retrieval-ablation.md`](retrieval-ablation.md) (`make ablate`).

| Change | Recall@1 | Recall@10 | MRR | Queries with misses @10 |
|---|---|---|---|---|
| BM25 baseline | 0.57 | 0.93 | 0.75 | 2 |
| + dense bge-small, fused with RRF ([ADR-005](../adr/0005-hybrid-retrieval.md)) | 0.70 | 0.97 | 0.87 | 1 |
| + cross-encoder rerank ([ADR-006](../adr/0006-reranking-and-ablation.md)) | **0.80** | **1.00** | **0.97** | **0** |

## Verifier - 21 labelled pairs (13 unsupported, 8 supported)

Source: [`verifier-calibration.md`](verifier-calibration.md) (`make calibrate`):
the NLI-alone sweep and the layer-by-layer table. False accept = an
unsupported bullet kept; false reject = a supported one dropped.

| Change | False accept | False reject |
|---|---|---|
| NLI alone at 0.80 (the first default) | 31% | 0% |
| Threshold 0.80 -> 0.95 ([ADR-009](../adr/0009-entailment-verifier.md), amended) | 15% | 0% |
| + number check ([ADR-008](../adr/0008-hard-grounding.md)) | 15% | 0% |
| + claim-strength check (ADR-008, amended) = the production gate | **0%** | **0%** |

What the rows say:

- The threshold change halved false accepts at no cost in false rejects.
- The number check catches nothing on this set that NLI had not already
  rejected. It stays: it is cheap, deterministic, and explains its drops.
- The claim-strength check removes the remaining false accepts ("led",
  "architected" scored 0.98-0.99 entailment). Caveat: its lexicon was
  written after seeing those failures, so this 0% is in-sample.

## Golden set - pending

137 bullets from two models are collected (`benchmarks/golden/golden.jsonl`);
human labels are still to do (ADR-010). Output fabrication rate will be
added here once they are.

## Adding a row

Make one change, rerun the benchmark it affects (`make ablate`,
`make calibrate` or `make eval`), commit the regenerated result file, and add
the row here with the number it produced.
