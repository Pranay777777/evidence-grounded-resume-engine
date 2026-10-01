---
title: Evidence-Grounded Resume Engine
emoji: 🧾
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Resume bullets that must cite verified evidence
---

Every bullet must cite a verified evidence record and pass a grounding gate
(citation checks, numbers, claim strength, NLI entailment). Unsupported
claims are dropped, never rewritten. The career shown here is synthetic.

Source, architecture, ADRs and measured results:
https://github.com/Pranay777777/evidence-grounded-resume-engine

Space secrets: `OPENROUTER_API_KEY` (free key) and optionally `LLM_MODEL`
(a pinned free model id). Without a key, the samples still work - they
replay recorded drafts through the live gate with no model call.
