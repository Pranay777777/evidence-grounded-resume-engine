# ADR-019: A draft UI that shows its evidence and its edits

- **Status:** accepted
- **Date:** 2026-10-01

## Decision

`/ui` (local mode only, like the admin) drafts from a pasted job description
and renders:

- every kept bullet with **inline numbered citations**; hovering or
  keyboard-focusing a citation shows the record - ID, revision, how it was
  verified, the statement, the project;
- a **base-vs-tailored diff** per bullet: the cited statements against the
  bullet, word by word, with additions and removals highlighted and the share
  of words not in the evidence. The gate decides what is kept; the diff shows
  a reader *how* each kept bullet differs from its evidence;
- dropped bullets with the gate's reason.

Server-rendered with Jinja autoescaping and the admin's CSRF protection; no
JavaScript build. The diff is built as data and rendered by the template, so
no generated text is ever marked safe.
