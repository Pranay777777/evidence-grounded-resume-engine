# Verifier calibration — 2026-09-28

Verifier `nli-deberta` on 21 labelled pairs (13 unsupported, 8 supported) from `benchmarks/verifier/pairs.yaml`. Three pairs are verbatim bullets from the first live generation; the rest are synthetic.

| Threshold | False accept | False reject | Accuracy |
|---|---|---|---|
| 0.50 | 38% | 0% | 76% |
| 0.60 | 31% | 0% | 81% |
| 0.70 | 31% | 0% | 81% |
| 0.80 | 31% | 0% | 81% |
| 0.90 | 23% | 0% | 86% |
| 0.95 | 15% | 0% | 90% |

## Per pair

| Pair | Source | Supported | Verdict | Entailment |
|---|---|---|---|---|
| real-cicd | real-draft | no | neutral | 0.01 |
| real-developed | real-draft | no | entailment | 0.56 |
| real-delta | real-draft | no | neutral | 0.00 |
| syn-faithful-ci | synthetic | yes | entailment | 0.98 |
| syn-faithful-pii | synthetic | yes | entailment | 0.99 |
| syn-faithful-runtime | synthetic | yes | entailment | 0.99 |
| syn-faithful-airflow | synthetic | yes | entailment | 0.99 |
| syn-faithful-dbt | synthetic | yes | entailment | 0.98 |
| syn-faithful-mentoring | synthetic | yes | entailment | 0.98 |
| syn-faithful-react | synthetic | yes | entailment | 0.99 |
| syn-led | synthetic | no | entailment | 0.98 |
| syn-whole-team | synthetic | no | entailment | 0.91 |
| syn-architected | synthetic | no | entailment | 0.99 |
| syn-kubernetes | synthetic | no | neutral | 0.00 |
| syn-realtime | synthetic | no | entailment | 0.86 |
| syn-cd | synthetic | no | neutral | 0.11 |
| syn-outcome | synthetic | no | neutral | 0.00 |
| syn-adoption | synthetic | no | neutral | 0.00 |
| syn-contradict-number | synthetic | no | contradiction | 0.00 |
| syn-contradict-direction | synthetic | no | contradiction | 0.00 |
| syn-two-records | synthetic | yes | entailment | 0.99 |
