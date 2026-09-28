# ADR-006: Cross-encoder reranking, measured by a published ablation

- **Status:** accepted
- **Date:** 2026-09-28

## Context

Hybrid retrieval (ADR-005) maximises recall: the right record is usually in
the top twenty. The generator reads the top few. Precision at the top is what
decides which evidence a résumé is built from.

## Decision

**A cross-encoder reranks the fused top 20.** Retrievers score the query and
a record separately; a cross-encoder reads them together, so it can separate
"built dashboards" from "used dashboards". It is too slow for the whole
store, so it runs only on fused candidates. The model is
`Xenova/ms-marco-MiniLM-L-6-v2` via fastembed — 80 MB, Apache-2.0, ONNX,
CPU — in the same `[embeddings]` extra as the embedder.

**Every configuration is measured, and the table is published.**
`python -m grounded.retrieval ablate` runs labelled queries through BM25,
dense (hashing and bge-small), hybrid, and hybrid + rerank, reporting
recall@1/3/5/10 and MRR, plus every miss at k = 10. A configuration whose
model is unavailable is listed as skipped with its reason — never replaced by
a different model under the same name.

**The shipped benchmark is synthetic, and says so.** Twenty-five records of a
fictional career and twenty queries, labelled by the author before any
configuration was run, mixing exact tool names with paraphrases. It compares
configurations; it is not evidence of quality on real data. That is the
golden set's job (step 50), which is human-labelled and real.

## Consequences

**Gained:** reranking is justified by a number, not a belief; a regression in
any configuration shows up as a changed row; misses are listed so a failure
can be read, not just counted.

**Given up:** a synthetic benchmark written by the same author as the system
risks flattering it; the per-query misses and the pre-run labelling are the
mitigation, and the golden set is the fix. The bge and reranker rows come from
a run on the developer's machine, because the build environment could not
download models. Reranking adds per-request latency proportional to the
candidate count.
