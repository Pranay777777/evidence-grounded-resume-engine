"""Claim strength: ownership and leadership words the evidence must state itself.

Calibration showed the NLI verifier's blind spot precisely: it scored
"Led the migration" as entailed by "Contributed to the migration" (0.98),
"Architected the backend" by "Wrote endpoints" (0.99), "Owned on-call" by
"Shared the on-call rotation" (0.91). The rest of the sentence overlaps, so
the model reads the verb upgrade as agreement. Scope inflation is also the
embellishment interviewers probe hardest.

So these terms are checked deterministically: if a bullet uses one, a cited
statement must use a word from the same group. The lexicon is deliberately
small and unambiguous — "architecture", "orchestrate" and "directed" are
left out because they have honest technical meanings — and a term flagged
here drops the bullet exactly as a failed number check does (ADR-008).
"""

from __future__ import annotations

import re

GROUPS: dict[str, tuple[str, ...]] = {
    "led": ("lead", "leads", "leading", "led"),
    "owned": ("owned", "owns", "owner", "ownership"),
    "architected": ("architect", "architects", "architected", "architecting"),
    "spearheaded": ("spearhead", "spearheads", "spearheaded", "spearheading"),
    "headed": ("headed",),
    "managed": ("managed", "managing", "manager", "manages"),
    "oversaw": ("oversaw", "oversee", "oversees", "overseeing", "overseen"),
    "sole": ("sole", "solely", "single-handedly", "singlehandedly"),
    "founded": ("founded", "co-founded", "cofounded"),
    "pioneered": ("pioneered", "pioneering"),
    "championed": ("championed",),
    "real-time": ("real-time", "real time", "realtime"),
}

_PATTERNS = {
    group: re.compile(
        r"(?<![\w-])(" + "|".join(re.escape(f) for f in forms) + r")(?![\w-])", re.IGNORECASE
    )
    for group, forms in GROUPS.items()
}


def claimed(text: str) -> set[str]:
    """The strength groups a text uses."""
    return {group for group, pattern in _PATTERNS.items() if pattern.search(text)}


def escalations(bullet: str, evidence_texts: list[str]) -> set[str]:
    """Strength terms in the bullet that no cited statement supports."""
    supported: set[str] = set()
    for text in evidence_texts:
        supported |= claimed(text)
    return claimed(bullet) - supported
