# Model comparison - 2026-09-29

Every model answered the same 20 synthetic job descriptions (`benchmarks/golden/jds.yaml`) from the same retrieved evidence; every bullet went through the production gate (checks + `nli-deberta` >= 0.95). Quality columns need no human labels; fabrication does.

| Model | Jobs | Bullets | Kept by gate | Citation P / R | Keyword coverage | Tone | Fabrication (labelled) | Tokens / draft | Latency / draft | Cost / draft |
|---|---|---|---|---|---|---|---|---|---|---|
| cohere/north-mini-code:free | 20 | 77 | 33 (43%) | 74% / 45% | 40% | 97% | unlabelled | - | - | $0.00 |
| nvidia/nemotron-3-super-120b-a12b:free | 19 | 60 | 34 (57%) | 90% / 57% | 43% | 97% | unlabelled | - | - | $0.00 |
| poolside/laguna-s-2.1:free | 17 | 76 | 37 (49%) | 49% / 38% | 40% | 95% | unlabelled | 1,150 | 7.8 s | $0.00 |

## Why the gate dropped bullets

- **cohere/north-mini-code:free** - entailment 16, neutral 28
- **nvidia/nemotron-3-super-120b-a12b:free** - entailment 15, neutral 11
- **poolside/laguna-s-2.1:free** - entailment 10, neutral 29

Tokens, latency and cost come from `benchmarks/golden/runs.jsonl`; `-` means the model's bullets were collected before runs were recorded. Cost is $0 for free-tier models and not estimated for paid ones.
