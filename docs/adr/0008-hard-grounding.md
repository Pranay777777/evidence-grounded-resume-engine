# ADR-008: Hard grounding — deterministic checks before any model

- **Status:** accepted, amended 2026-09-28 (claim strength)
- **Date:** 2026-09-28

## Context

ADR-001 requires every bullet to cite verified evidence, and ADR-007 makes
citations structural: a bullet without `evidence_ids` does not parse. But a
parsed citation proves nothing. The first live draft (2026-09-28) cited three
real, verified records — and every bullet still said more than its record.

## Decision

Before the entailment model sees a bullet, four deterministic checks run,
cheapest first. The first failure drops the bullet with a recorded reason:

1. **The citation resolves.** Every cited ID was among the records given to
   the model. A model can invent an ID as easily as a fact.
2. **The citation is current.** Each record is still verified, at the
   revision that was retrieved. A record edited mid-request cannot lend its
   old verification to a new claim (ADR-002).
3. **Every number is supported.** Each quantity in the bullet must appear in
   a cited record's statement, metric value or month. Number words and
   magnitudes are normalised — "five" is 5, "1.6 million" and "1.6M" are
   1,600,000 — so honest restatements pass and inflated figures do not. A
   test shows "3 million rows" failing even though a bare 3 appears in the
   evidence ("3 GB"): magnitude matters.
4. **Only then, entailment** (ADR-009).

**Dropped means dropped.** The report carries the bullet exactly as the
model wrote it, and the reason. Nothing is paraphrased, softened or sent
back: regenerating until a check passes optimises for the check (ADR-001).

**No silent skip.** Without the entailment model a draft can still be
produced, but only with an explicit `--unverified` flag, and the output then
says UNVERIFIED on its first and last lines.

## Consequences

**Gained:** the most common fabrication — an invented or inflated number —
is caught without a model, deterministically, before any inference cost;
stale and invented citations can never survive.

**Given up:** the number check is strict. A bullet that derives a number
("doubled") or states a year the record does not is dropped, even when true.
That is the preferred direction of error. Technology names, scope and
outcomes are not checkable this way; they are what ADR-009 is for.

## Amendment — claim strength

The first calibration run showed the NLI verifier's blind spot exactly: it
scored "Led the migration" as entailed by "Contributed to the migration"
(0.98), "Architected the backend" by "Wrote endpoints" (0.99), "Owned
on-call" by "Shared the on-call rotation" (0.91), and "real-time" by
"nightly" (0.86). The rest of each sentence overlaps, so the verb upgrade
reads as agreement — and scope inflation is precisely what interviewers
probe.

A fifth deterministic check now runs before NLI: ownership and leadership
terms (led, owned, architected, spearheaded, managed, sole, founded, …) and
"real-time" must appear, in some form, in a cited statement. The lexicon is
small and grouped by root; words with honest technical meanings
("architecture", "orchestrate", "directed") are deliberately excluded.

**Caveat, recorded before the re-run:** the lexicon was written after seeing
those failures, so the calibration set is now partly its training data.
Replaying the committed NLI verdicts predicts the full gate at 0% false
accept and 0% false reject on those 21 pairs; that figure is optimistic by
construction. The independent test is the human-labelled golden set (step
50), which the lexicon has never seen.
