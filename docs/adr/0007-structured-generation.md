# ADR-007: Structured generation through OpenRouter, retrying only structure

- **Status:** accepted, amended 2026-09-28 (see *Amendment*)
- **Date:** 2026-09-28

## Context

Retrieved evidence has to become résumé bullets, each carrying the IDs of the
records it relies on (ADR-001, rule 2). That needs a model, a transport, an
output contract, and a policy for when the model gets it wrong.

## Decision

### Provider: OpenRouter, with a pinned free model

OpenRouter exposes many models behind one OpenAI-compatible endpoint, with
free variants — no paid key needed to build and demo. The default is
`openai/gpt-oss-20b:free`, which supports function calling and structured
output. It is **pinned**, not the `openrouter/free` router: that router picks
a model at random per request, and an evaluation run against a different
model each time measures nothing (step 50 depends on this).

**Privacy caveat, stated plainly:** on free models, the provider may log
prompts and completions and use them to improve its models. The evidence sent
is résumé content — facts a candidate intends to publish — but it is still
personal data leaving the machine. Step 58 adds PII redaction and a local-model
mode; until then, keep confidential employer detail out of the evidence store.

### A small client, not an SDK

About a hundred lines over `httpx`: one endpoint, trivially mocked, explicit
about retries. Free tiers rate-limit hard, so 429 and 5xx responses back off
exponentially, honouring `Retry-After`; 4xx errors fail at once, because
retrying cannot fix them. The API key lives in a `SecretStr` and only ever in
the Authorization header — a test asserts it appears in no exception text.
Step 55 puts more providers behind the same interface.

### Output contract: a forced tool call, validated by Pydantic

The prompt forces a call to `emit_draft`, whose JSON schema is generated from
the `Draft` model: one to eight bullets, each 15–300 characters with one to
four `evidence_ids`. Models that answer in plain content instead are still
accepted — the JSON is taken from the content, stripped of code fences, and
validated identically.

### Retries fix structure, never facts

If the output is not valid JSON, or fails the schema, the model is shown its
own output and the validation errors and asked again — at most three
attempts. This is the only retry ADR-001 permits. Output that parses but
cites the wrong evidence is a *grounding* failure: it is never sent back,
because regenerating until a checker is satisfied optimises for the checker.
Steps 48 and 49 drop such bullets instead.

### The job description is untrusted data

It is fenced in `<job_description>` tags, capped at 20,000 characters, and
the system prompt states — before the model sees it — that nothing inside
the tags is an instruction. That is a first line only; the grounding checks
decide what survives regardless of what the model was talked into, and
step 57 red-teams the prompt. The prompt is version-stamped
(`generate-v1`) so results can be tied to the prompt that produced them.

## Consequences

**Gained:** generation that never returns an unparseable or uncited bullet;
a free, reproducible default model; a client whose failure modes are tested.

**Given up:** free models are slower, rate-limited and may disappear from the
catalogue — the model is one setting to change. Output is a *draft* until
steps 48–49 check every citation. The build environment could not reach
OpenRouter, so the client is tested against a mock transport and exercised
for real only on the developer's machine.

## Amendment — the pinned free model stopped being free

Within a day of this decision, OpenRouter withdrew the free variant of
`openai/gpt-oss-20b` (the API now answers 404, "unavailable for free"). The
risk named above — free models disappear from the catalogue — was real and
fast.

- **Drafts now default to `openrouter/free`**, OpenRouter's router, which
  always selects an available free model that supports the request's
  features, tool calling included. Every draft prints the model it actually
  got, so no output is ever unattributed.
- **`python -m grounded.generation models`** lists, live from the public
  catalogue, the free models that support tool calling — so a pin is chosen
  from what exists today, not from memory.
- **Evaluation runs still require a pinned model.** The router's random
  choice is acceptable for a draft and meaningless for a benchmark; the eval
  harness (step 51) will refuse `openrouter/free`.
