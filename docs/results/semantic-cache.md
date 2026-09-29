# Semantic cache benchmark - 2026-09-29

24 labelled job-description pairs (12 the same request, 12 different) from `benchmarks/cache/pairs.yaml`; embedder `bge-small`. No model was called.

| Threshold | Hit rate (similarity) | False hits (similarity) | Hit rate (+ same evidence) | False hits (+ same evidence) |
|---|---|---|---|---|
| 0.80 | 100% | 25% | 50% | 0% |
| 0.85 | 100% | 17% | 50% | 0% |
| 0.90 (configured) | 75% | 0% | 50% | 0% |
| 0.95 | 50% | 0% | 42% | 0% |
| 0.97 | 33% | 0% | 33% | 0% |
| 0.99 | 33% | 0% | 33% | 0% |

The cache's rule is the right-hand pair of columns.

Savings at 0.9: if every request were a repeat or rephrase, 50% of them hit, saving about 57,491 tokens per 100 requests (mean 1,150 tokens per draft) = $0.0000 at $0.0/M tokens (free models cost $0).

## Per pair

| Pair | Kind | Same request | Similarity | Same evidence |
|---|---|---|---|---|
| fmt-airflow | format | yes | 1.000 | yes |
| fmt-streaming | format | yes | 1.000 | yes |
| fmt-ml | format | yes | 1.000 | yes |
| fmt-web | format | yes | 1.000 | yes |
| para-batch | paraphrase | yes | 0.958 | yes |
| para-quality | paraphrase | yes | 0.895 | no |
| para-streaming | paraphrase | yes | 0.963 | no |
| para-rag | paraphrase | yes | 0.874 | no |
| para-mlops | paraphrase | yes | 0.945 | no |
| para-platform | paraphrase | yes | 0.948 | yes |
| para-privacy | paraphrase | yes | 0.851 | no |
| para-cost | paraphrase | yes | 0.913 | no |
| diff-stack-batch | different stack | no | 0.864 | no |
| diff-stack-web | different stack | no | 0.734 | no |
| diff-role-ml | different role | no | 0.763 | no |
| diff-role-rag | different role | no | 0.730 | no |
| diff-focus-quality | different focus | no | 0.771 | no |
| diff-focus-streaming | different focus | no | 0.779 | yes |
| diff-focus-ml | different focus | no | 0.809 | no |
| diff-focus-security | different focus | no | 0.864 | no |
| diff-lead | different seniority ask | no | 0.730 | no |
| diff-privacy | different focus | no | 0.791 | no |
| diff-unrelated | unrelated | no | 0.583 | no |
| diff-unrelated-2 | unrelated | no | 0.634 | no |
