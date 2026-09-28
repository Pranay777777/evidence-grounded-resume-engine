# ADR-004: One record per chunk; bge-small via ONNX; embeddings bound to revisions

- **Status:** accepted
- **Date:** 2026-09-28

## Context

Dense retrieval needs text split into chunks and an embedding model. Both
choices interact with ADR-001 and ADR-002: whatever is retrieved is what the
generator may cite, and a citation must point at exactly one verified fact.

## Decision

### Chunking: one record is one chunk

ADR-002 already made every record a single checkable fact of at most 400
characters. The record boundary is the citation boundary, so it is also the
chunk boundary:

- **Splitting** a record would hand the generator half a fact to cite.
- **Merging** records — the usual fixed-window or paragraph chunking —
  would let a bullet cite one fact while silently leaning on its neighbour,
  which is exactly the gap the entailment verifier exists to close.
- **Sliding windows and overlap** solve a problem long documents have and
  atomic records do not.

The text embedded for a record is its statement followed by its project
(name and summary), role, and skills. A statement like "the suite has 499
tests" retrieves poorly for "Python data platform" on its own, because the
words that matter live on the project. Only metadata the record already
carries is added — never a fact it does not state.

### Model: BAAI/bge-small-en-v1.5 through fastembed

384 dimensions, 67 MB, MIT-licensed, strong for its size on retrieval
benchmarks, and run through ONNX Runtime by `fastembed` — no PyTorch, CPU
only, installs cleanly on Windows. It is an optional extra
(`pip install -e ".[embeddings]"`), so the default install stays small.

A second embedder, `hashing`, maps word unigrams and bigrams into the same
384 dimensions by feature hashing. It needs no model and no network, which
makes it what the tests and CI use. It is reported in the retrieval ablation
(step 46) for what it is: a lexical baseline in vector form, not a semantic
model. Asking for `bge-small` without the extra raises an error with the
install command — never a silent fallback to `hashing`, which would make
every retrieval number reported afterwards misattributed.

### Storage: per record, per embedder, per revision

`evidence_embedding` holds one vector per (record, embedder), stamped with
the record revision it encodes. On Postgres the column is pgvector's
`vector(384)` with an HNSW cosine index; on SQLite, used by unit tests, the
same list is stored as JSON. Only citable records are embedded. A record
edited since its vector was made is stale and excluded from dense search
until it is embedded again — the vector describes a statement that no
longer exists.

## Consequences

**Gained:** retrieval can never return anything but a whole, verified,
current fact; two models can be compared on the same corpus; the default
install and CI need no model download.

**Given up:** the bge model was not exercised in the environment this was
built in (no access to its host), only on the developer's machine — the
adapter is covered by a test double, and its real behaviour by the smoke
test in the step's instructions. A fixed 384 dimensions means a larger
model later needs a migration. Context lines inflate every record's text
equally, so they help recall but do not discriminate between records of
the same project; BM25 and reranking have to do that.
