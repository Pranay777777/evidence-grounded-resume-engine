# Architecture decision records

One short file per decision that would be expensive to reverse, numbered in
order, following [`0000-template.md`](0000-template.md). The reasoning lives
next to the code, so a reviewer can challenge the decision rather than
reverse-engineer it — and so a decision is changed on purpose, not by
accident.

| ADR | Decision |
|---|---|
| [001](0001-grounding-constraint.md) | The grounding constraint: unsupported claims are rejected, never smoothed over |
| [002](0002-evidence-model.md) | Atomic, versioned evidence records; a changed fact is an unverified fact |
| [003](0003-evidence-api-and-admin.md) | One write path for evidence, a local CSRF-protected admin, no deletes |
