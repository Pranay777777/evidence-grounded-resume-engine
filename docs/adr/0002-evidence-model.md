# ADR-002: Atomic, versioned evidence records

- **Status:** accepted
- **Date:** 2026-09-28

## Context

ADR-001 makes evidence the only source of fact. That puts all the weight on
what an evidence record *is*: if records are vague, compound, or silently
editable, grounding is theatre — a bullet can cite a record that no longer
says what it said when it was checked.

## Decision

### One checkable fact per record

A record's `statement` is a single claim a reader could verify. If it needs
"and", it is two records. The entailment verifier (step 49) checks each
bullet against the records it cites; a compound record lets a bullet lean
on the half that is true while stating the half that is not.

### Three tables, and only evidence is citable

`role` and `project` give context for retrieval and phrasing. `evidence`
holds the facts. A bullet cites evidence IDs only — "worked at X" is a fact
once an evidence record states it, not because a role row exists.

### Stable IDs, and revisions that void verification

IDs are human-readable slugs, immutable once used, because citations point
at them. Every record carries a `revision` and a hash of the fields that
make up the fact. **When the hash changes, the revision increments and the
verification is cleared** — whatever the file says. Verification attests to
a specific statement, so editing the statement voids it. Without this, a
record verified as "contributed to the migration" could be edited to "led
the migration" and remain citable. Generations will record the revision
they cited, so an edit can never retroactively change what a past résumé
claimed. Skills tags are excluded from the hash: retagging is not a new fact.

### Verification says how, not just whether

`verified` requires a method, shown with every citation: `artifact` (a
public link, which must be present), `third_party` (a named confirmer), or
`self_attested` (the candidate's word, labelled as such). Only `verified`
records are citable. `rejected` records are kept, so a disproved claim is on
the record and cannot quietly come back.

### Numbers live only in metric records

A record carries `metric_value` if and only if its kind is `metric`,
enforced by a CHECK constraint. The generator may quote a number only from
a metric record — the single most common fabrication in a résumé is a
plausible percentage, and this makes every number traceable to one place.

### Months, not days

Dates are `YYYY-MM`. Résumés are written at month precision, and a day
would imply knowledge the evidence rarely has.

### YAML in, database out

People write evidence as YAML — reviewable in a diff, easy to keep in
version control — and the loader validates it with messages a person can
act on, then upserts by ID. The database constraints are a second line, not
the first. A load never deletes: a record missing from the file stays, so a
typo cannot erase history.

### Private by default

Real evidence goes in `evidence/private/`, which is gitignored. Employer
work may be confidential regardless of being true. Records backed by
artifacts that are already public can be committed under `evidence/drafts/`.

## Options considered

1. **Free-text career history, chunked for retrieval.** The easiest to fill
   and the hardest to verify: a chunk is not a claim, and a bullet citing a
   paragraph can be "supported" by any sentence in it.
2. **Mutable records without revisions.** Simpler, and it breaks the chain
   between a verification and the statement it verified.
3. **A graph of skills, roles and projects.** Richer for retrieval, but the
   generator cites facts, not edges, and the extra structure buys nothing
   the verifier can check.

## Consequences

**Gained:** every citation resolves to one short, individually verified,
versioned statement; a verification cannot outlive the fact it checked;
every number has one source.

**Given up:** writing evidence is slower — splitting compound achievements
into atomic records is real work, and some candidates will find the
discipline tedious. The pgvector extension is enabled in the baseline
migration, but embeddings are deferred to step 44, where the chunking and
model decisions set the vector dimension.
