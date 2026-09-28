# ADR-005: Hybrid retrieval — BM25 and dense, fused by reciprocal rank

- **Status:** accepted
- **Date:** 2026-09-28

## Context

The generator can only cite what retrieval hands it. A relevant record that
is never retrieved is a bullet that is never written; an irrelevant one
wastes a slot and invites a strained citation.

## Decision

**Two retrievers, because they fail differently.** BM25 matches exact
terms — `HNSW`, `Delta Lake`, `c++` — that embeddings blur into their
neighbours. The dense model matches meaning — "data platform" for a record
about a lakehouse — with no shared words. Job descriptions need both: they
mix precise technology names with loose descriptions of the work.

**Reciprocal rank fusion, not score blending.** Each record's fused score
is Σ 1/(60 + rank) over the retrievers that returned it. RRF uses only
ranks, so BM25's unbounded scores and cosine similarities never have to be
put on one scale, and there is no weight to tune per corpus. k = 60 is the
value from Cormack et al. (2009); it rarely loses to tuned alternatives.
Each retriever contributes its top 50, wider than the final k, so a record
ranked moderately by both can outrank one ranked highly by only one.

**BM25 in-process, dense in Postgres.** The evidence corpus is hundreds of
short records, so an in-memory Okapi BM25 (k1 = 1.5, b = 0.75) over the
citable set is exact and fast; a search engine would be infrastructure
without a benefit. Dense search runs in Postgres through pgvector's `<=>`
cosine operator and the HNSW index — a test checks its ordering matches a
plain cosine computation, and that the plan uses the index.

**The tokeniser keeps technical terms whole.** `c++`, `c#`, `node.js` and
`3.12` survive as single tokens. An early version split `c++` into `c`,
which a test caught — BM25 would have been blind to precisely the terms job
descriptions are made of.

**The query is untrusted and only ever search text.** A job description is
tokenised for BM25 and embedded for dense search — never interpreted — and
capped at 20,000 characters so an oversized input cannot turn retrieval
into a denial of service.

**Every hit records where each retriever ranked it.** That is what the
reranking ablation in step 46 is built from.

## Consequences

**Gained:** exact-term and semantic recall in one ranked list, with no
weights to tune and a per-hit record of why it was retrieved.

**Given up:** dense search always ranks every record, so retrieval is not a
relevance filter — an unrelated query still returns k results, and it is the
reranker (step 46) and the entailment verifier (step 49) that must refuse to
use a poor match. Quality is not yet measured; recall@k for BM25, dense,
hybrid and reranked configurations is step 46's published table.
