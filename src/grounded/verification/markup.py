"""Reject bullets that carry markup (OWASP LLM10:2026, improper output handling).

A résumé bullet is one plain sentence. HTML tags become script injection when
a bullet is rendered, Markdown images become an exfiltration channel when a
client auto-fetches them, and terminal escape sequences rewrite what a reader
sees. None of these can be supported by evidence, but an entailment model
judges meaning, not format: "<script>...</script> Wrote Airflow DAGs." can
still be entailed by a record about Airflow. So format is checked here,
deterministically, before the model is asked anything. A URL is allowed only
if the cited evidence itself contains it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

CHECKS: dict[str, re.Pattern[str]] = {
    "an HTML tag": re.compile(r"</?[A-Za-z][^>]*>"),
    "a Markdown link or image": re.compile(r"!?\[[^\]]*\]\([^)]*\)"),
    "a control character": re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x9b]"),
    "a code fence": re.compile(r"```|`[^`]+`"),
}
URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\"')\]]+", re.IGNORECASE)


def markup(text: str, facts: Sequence[str]) -> list[str]:
    """What kinds of markup `text` carries that the evidence does not."""
    found = [kind for kind, pattern in CHECKS.items() if pattern.search(text)]
    evidence = " ".join(facts).casefold()
    if any(m.group(0).casefold().rstrip(".,;:") not in evidence for m in URL.finditer(text)):
        found.append("a URL not in the evidence")
    return found
