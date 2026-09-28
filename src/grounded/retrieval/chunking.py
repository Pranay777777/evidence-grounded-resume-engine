"""What gets embedded for each evidence record — and why it is one chunk.

**One record is one chunk.** ADR-002 already made every record a single
checkable fact of at most 400 characters. Splitting a record would give a
bullet half a fact to cite; merging records would let a bullet cite one
fact while leaning on its neighbour. The record boundary *is* the citation
boundary, so it is also the chunk boundary.

**The embedded text adds context, never facts.** A statement like "The test
suite has 499 tests" retrieves poorly for "Python data platform" because the
words that matter live on the project, not the statement. So the text that
is embedded is the statement followed by its project, role and skills —
metadata the record already carries. Nothing is added that the record does
not state, so retrieval can never surface a record for a reason the
verifier would not accept.
"""

from __future__ import annotations

from grounded.evidence.models import Evidence


def document_text(record: Evidence) -> str:
    """The text embedded and BM25-indexed for one record."""
    lines = [record.statement.strip()]
    if record.project is not None:
        summary = f" — {record.project.summary}" if record.project.summary else ""
        lines.append(f"Project: {record.project.name}{summary}")
    role = record.role or (record.project.role if record.project else None)
    if role is not None:
        lines.append(f"Role: {role.title} at {role.organisation}")
    if record.skills:
        lines.append("Skills: " + ", ".join(record.skills))
    return "\n".join(lines)
