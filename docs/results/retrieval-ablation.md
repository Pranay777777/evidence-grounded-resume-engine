# Retrieval ablation — 2026-09-28

Synthetic benchmark: 20 labelled queries over a fictional career (`benchmarks/retrieval/`), labelled by the author before any run. It compares configurations; it is not the golden set.

| Configuration | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| BM25 | 0.57 | 0.80 | 0.82 | 0.93 | 0.75 |
| Dense / hashing | 0.47 | 0.57 | 0.72 | 0.93 | 0.63 |
| Dense / bge-small | 0.60 | 0.95 | 0.95 | 0.97 | 0.83 |
| Hybrid / hashing | 0.57 | 0.80 | 0.82 | 0.93 | 0.74 |
| Hybrid / bge-small | 0.70 | 0.88 | 0.90 | 0.97 | 0.87 |
| Hybrid / bge-small + rerank | 0.80 | 0.93 | 0.95 | 1.00 | 0.97 |

## Misses at k = 10

- **BM25** — q19: b-oncall, b-docs; q20: b-fastapi-serving
- **Dense / hashing** — q19: b-oncall, b-docs; q20: b-fastapi-serving
- **Dense / bge-small** — q20: b-fastapi-serving
- **Hybrid / hashing** — q19: b-oncall, b-docs; q20: b-fastapi-serving
- **Hybrid / bge-small** — q20: b-fastapi-serving
- **Hybrid / bge-small + rerank** — none
