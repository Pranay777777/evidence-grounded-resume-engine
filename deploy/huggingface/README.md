---
title: Evidence-Grounded Resume Engine
emoji: 🧾
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.29.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Resume bullets that must cite verified evidence
---

Every bullet must cite a verified evidence record and pass a grounding gate
(citation checks, numbers, claim strength, NLI entailment). Unsupported
claims are dropped, never rewritten. The career shown here is synthetic.

The Space installs the tagged release and runs the project's own server; the
Gradio SDK is only the runtime. The first start downloads the embedding and
NLI models, so it takes a few minutes.

Source, architecture, ADRs and measured results:
https://github.com/Pranay777777/evidence-grounded-resume-engine

Space secret: `OPENROUTER_API_KEY` (free key). Optional variable: `LLM_MODEL`
(a pinned free model id). Without a key, the samples still work - they
replay recorded drafts through the live gate with no model call.
