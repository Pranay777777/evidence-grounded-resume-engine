"""Word-level diff between what the evidence says and what the bullet says.

The "base vs tailored" view (step 62): for each kept bullet, the cited
evidence statements are the base and the bullet is the tailored text. The
diff shows exactly which words the model added or dropped while tailoring -
so embellishment is visible at a glance, even in bullets the gate kept.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Literal

Op = Literal["same", "added", "removed"]


def _key(word: str) -> str:
    return word.strip(".,;:()\"'").casefold()


def word_diff(base: str, tailored: str) -> list[tuple[Op, str]]:
    """Runs of (op, words) turning `base` into `tailored`, words joined by single
    spaces. Matching ignores case and surrounding punctuation, so "Built" and
    "built," count as the same word."""
    a, b = base.split(), tailored.split()
    runs: list[tuple[Op, list[str]]] = []

    def emit(op: Op, words: list[str]) -> None:
        if not words:
            return
        if runs and runs[-1][0] == op:
            runs[-1][1].extend(words)
        else:
            runs.append((op, list(words)))

    matcher = SequenceMatcher(a=[_key(w) for w in a], b=[_key(w) for w in b], autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            emit("same", b[j1:j2])
        else:
            emit("removed", a[i1:i2])
            emit("added", b[j1:j2])
    return [(op, " ".join(words)) for op, words in runs]


def added_share(diff: list[tuple[Op, str]]) -> float:
    """Share of the tailored text's words that are not in the base."""
    added = sum(len(t.split()) for op, t in diff if op == "added")
    total = sum(len(t.split()) for op, t in diff if op in ("same", "added"))
    return added / total if total else 0.0
