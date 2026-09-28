# ADR-001: The grounding constraint

- **Status:** accepted
- **Date:** 2026-09-28

## Context

A résumé generator has one failure that matters more than every other:
stating something that is not true. A model asked to tailor a résumé to a
job description is under constant pressure to fabricate — to turn "used
Databricks" into "led a Databricks migration", to invent a percentage, to
add the one skill the posting asks for. Each of those reads well and is a
lie the candidate then has to defend in an interview.

Most generation systems treat this as a tuning problem: a better prompt, a
lower temperature, an instruction to "only use the provided information".
Those reduce the rate. None of them make it measurable, and none of them
make it zero, because nothing checks the output.

## Decision

**Every claim this system emits must cite the evidence that supports it,
and a claim that its evidence does not support is rejected — never
rewritten until it sounds supported.**

Concretely:

1. **Evidence is the only source of fact.** A claim is anything a reader
   could check: a role, a date, a technology, a number, an outcome. Each
   one must trace to an *evidence record* — a stored, individually
   verified fact with a stable ID and, where one exists, an artifact that
   proves it.
2. **Citation is structural, not stylistic.** Every generated bullet
   carries the `evidence_ids` it relies on as data, validated against a
   schema. A bullet with no citations is invalid output, not a weak one.
3. **Citations are checked, not trusted.** A verifier decides whether the
   cited evidence *entails* the bullet. Citing a real record does not make
   a claim true; "led the migration" is not entailed by a record that says
   "contributed to the migration".
4. **Rejection means removal.** A bullet that fails verification is
   dropped, and the reason is recorded. It is never paraphrased, softened
   or hedged into passing, because a model rewriting a claim until a
   checker accepts it is optimising against the checker, not towards the
   truth.
5. **Shorter and true beats complete and false.** If rejection leaves a
   section thin, the section is thin. If nothing survives, the system
   returns nothing and says why. It never falls back to ungrounded
   generation.
6. **The failure rate is published.** Fabrication rate — claims not
   entailed by their evidence, over all claims, on a human-labelled golden
   set — is measured on every change and gated in CI. A number that is
   not measured is a number that is not known.

## Options considered

1. **Prompting for faithfulness.** "Only use the information provided."
   Cheap, and it helps, but it is a request rather than a constraint:
   nothing enforces it and nothing measures how often it is ignored.
2. **Retrieval-augmented generation without verification.** Putting the
   right evidence in context makes fabrication rarer, not impossible —
   models still embellish what they are given. Retrieval is necessary;
   on its own it is not sufficient.
3. **Rewrite-until-it-passes.** Feed verifier failures back to the model
   and regenerate. Tempting, and exactly the smoothing this ADR forbids:
   it produces claims tuned to pass the verifier, which is a different
   property from being true. Regeneration is allowed only for *structural*
   failures — output that does not parse — never for failed entailment.
4. **Human review of every output.** Accurate, and the reason a golden
   set exists, but it does not scale to every request and it hides the
   system's own error rate behind a person.
5. **Hard grounding with verification and rejection** (chosen).

## Consequences

**Gained:**
- A property that can be stated plainly and tested: every surviving claim
  names its evidence, and the evidence was checked.
- A published fabrication rate instead of an assurance.
- An honest failure mode — a thin résumé — in place of a dangerous one.

**Given up:**
- **Output that is sometimes less impressive.** Grounded bullets are
  bounded by what the evidence says, and a candidate with thin evidence
  gets a thin résumé. That is the point, and some users will dislike it.
- **Latency and cost.** Every bullet is verified, which is at least one
  more model call or inference per claim.
- **A verifier that can itself be wrong.** Entailment checking has false
  positives (letting an unsupported claim through) and false negatives
  (rejecting a true one). Both are measured against the golden set, and
  the verifier is tuned to prefer false negatives: a wrongly rejected true
  claim costs a bullet; a wrongly accepted false claim costs trust.
- **Evidence has to exist first.** The system is only as good as its
  evidence store, so populating and verifying it is real, unavoidable
  work, not an afterthought.

## What this rules out

Any feature whose value depends on saying more than the evidence supports:
inferring skills from job titles, estimating metrics the evidence does not
state, or "filling gaps" to match a job description. When a job asks for
something the evidence does not show, the correct output is to leave it
out — and, where useful, to tell the candidate what evidence would let them
claim it.
