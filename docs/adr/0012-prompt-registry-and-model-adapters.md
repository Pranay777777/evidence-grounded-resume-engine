# ADR-012: Versioned prompts in a registry; models behind one adapter spec

- **Status:** accepted
- **Date:** 2026-09-29

## Context

Every number this project publishes depends on two inputs besides the code:
the prompt and the model. Until now the prompt was a module constant with a
version string nobody enforced, and the model was always "whatever
OpenRouter serves". Providers deprecate models on published dates and free
tiers disappear overnight (ADR-007) - a provider swap has to be a config
change, and a result has to say exactly which prompt and model produced it.

## Decision

### Prompt registry

- Prompts live in `grounded.generation.prompt.REGISTRY`, keyed by version
  (`generate-v1`). Each has a **fingerprint** - a hash of its text.
- The fingerprints are **pinned in the tests**: editing a registered prompt
  in place fails CI. A change is a new version, so an old result can always
  be traced to the exact text behind it.
- The user message is a `string.Template` substituted once, so text inside
  the untrusted job description (`$evidence`, `${...}`) is never expanded.
- Every draft prints its prompt version and fingerprint; every golden-set
  item and collection run records them; eval reports list them. `--prompt`
  (or `PROMPT_VERSION`) chooses one; `python -m grounded.generation prompts`
  lists them.

### Model adapters

One spec string picks provider and model: an OpenRouter id (the default),
`ollama:<model>` (local - no key, nothing leaves the machine; the base of
step 58's local mode) or `openai:<model>`. All three speak the OpenAI
chat-completions protocol, so an adapter is only endpoint, credential and
headers; the grounding gate is identical whichever model wrote the draft.
A prefix counts only when it names a known provider, because OpenRouter ids
contain `:` themselves (`...:free`).

### Comparison

`python -m grounded.evals collect` now logs every call to
`benchmarks/golden/runs.jsonl` (tokens, latency, attempts), and
`python -m grounded.evals compare` reports, per model, on the same job
descriptions and the same evidence: bullets kept by the gate, drop reasons,
citation precision/recall, keyword coverage, tone - none of which need
labels - plus tokens, latency and cost per draft, and fabrication once
labelled. Cost is $0 on free tiers and is not estimated for paid models:
prices change, and a guessed price would be a made-up number.

## Consequences

- Bullets collected before this change have no fingerprint or run record;
  the comparison shows `-` for their tokens and latency.
- Adding a provider that does not speak the OpenAI protocol (a native SDK)
  would need a second client class behind the same `ChatClient` protocol.
