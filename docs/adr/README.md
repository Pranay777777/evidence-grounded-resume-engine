# Architecture decision records

One short file per decision that would be expensive to reverse, numbered in
order, following [`0000-template.md`](0000-template.md). The reasoning lives
next to the code, so a reviewer can challenge the decision rather than
reverse-engineer it — and so a decision is changed on purpose, not by
accident.

| ADR | Decision |
|---|---|
| [001](0001-grounding-constraint.md) | The grounding constraint: unsupported claims are rejected, never smoothed over |
| [002](0002-evidence-model.md) | Atomic, versioned evidence records; a changed fact is an unverified fact |
| [003](0003-evidence-api-and-admin.md) | One write path for evidence, a local CSRF-protected admin, no deletes |
| [004](0004-chunking-and-embeddings.md) | One record per chunk; bge-small via ONNX; embeddings bound to revisions |
| [005](0005-hybrid-retrieval.md) | Hybrid retrieval: BM25 + dense, fused by reciprocal rank |
| [006](0006-reranking-and-ablation.md) | Cross-encoder reranking, measured by a published ablation |
| [007](0007-structured-generation.md) | Structured generation through OpenRouter; retries fix structure, never facts |
