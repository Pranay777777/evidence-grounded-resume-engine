# A résumé generator that isn't allowed to make things up

## The problem

Ask a language model to tailor a résumé to a job description and it is
under pressure to invent. "Used Databricks" becomes "led a Databricks
migration". A percentage appears from nowhere. The one skill the posting
wants turns up in a bullet. Each line reads well, and each is something the
candidate has to defend in an interview. One invented metric, found out,
can cost an offer. Prompting for honesty lowers the rate. It does not make
it zero, and most tools never measure it.

## The rule

I wrote the rule down before any code, as
[ADR-001](adr/0001-grounding-constraint.md). Every claim cites a stored,
individually verified evidence record by its ID. A bullet without
citations is invalid output. A verifier then checks that the cited
evidence actually supports the bullet. If it doesn't, the bullet is
dropped, with the reason recorded. It is never rewritten and resubmitted
until it passes: that loop produces claims tuned to pass the verifier,
which is not the same as true. Shorter and true beats complete and false. If nothing
survives, the engine returns nothing and says why.

## How a claim gets checked

First the engine finds the right evidence. Retrieval is hybrid: BM25 and a
small embedding model, fused, then reranked by a cross-encoder
([ADR-005](adr/0005-hybrid-retrieval.md),
[ADR-006](adr/0006-reranking-and-ablation.md)). On a 20-query synthetic
benchmark, the right record now ranks first 80% of the time.

Then every generated bullet goes through a gate
([ADR-008](adr/0008-hard-grounding.md)). Each citation must be one of the
records the model was given, still verified at the revision retrieved. Cheap checks come next:
markup, numbers, and claim strength. Finally an NLI model reads the cited
evidence as the premise and the bullet as the hypothesis. Only entailment
at 0.95 or above passes; "neutral" fails, because neutral is what
embellishment looks like ([ADR-009](adr/0009-entailment-verifier.md)).

I set that threshold by measurement, not taste. I swept thresholds over 21
labelled pairs. NLI alone let 31% of unsupported bullets through at 0.80
and 15% at 0.95, and rejected no supported pair at any threshold. So the
stricter setting cost nothing measurable.

## Measuring it honestly

I labelled 110 generated bullets by hand, from three free models
([ADR-010](adr/0010-golden-set-and-evals.md)). Before the gate, 23% were
unsupported. After it, none of the 53 kept bullets were: 0 of 53, with a
95% upper bound of 7%. CI fails if a change lets one fabricated bullet
through ([results](results/eval.md)).

The cost is real. The gate also rejects 38% of supported bullets. That is
32 true bullets that a reader never sees. I chose that trade on purpose,
but it is a trade.

Not every change helped. The [improvement curve](results/improvement-curve.md)
records each change against the same benchmark. Adding the number check
left false accepts at 15%: it caught nothing NLI had not already caught. I
kept it because it is cheap and explains its drops. The claim-strength
check took false accepts to 0%, but I wrote its word list after seeing the
failures, so that 0% is in-sample. Separately, a
[red-team suite](results/redteam.md) of 34 hostile job descriptions assumes
the model obeyed each injection and checks that the gate still drops it.

## Using it on myself

My own one-page résumé is rendered by the engine from 25 verified records,
with no model involved. Each project bullet links to its public proof. It
is the same rule applied to its author: if I can't point to the proof, the
line isn't there. The
[live demo](https://huggingface.co/spaces/7Pranay77/evidence-grounded-resume-engine)
runs the gate on a synthetic career
([ADR-020](adr/0020-public-demo.md)).

## What I'd do differently

I would label the golden set earlier. Until I did, the headline number was
"pending". I would also hold out a set for the claim-strength rules before
writing them, so their result is not in-sample. And I would look at false
rejects as hard as false accepts, since 38% is a lot of true work left out.
