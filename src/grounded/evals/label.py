"""Label golden-set bullets by hand. Resumable: progress is saved after each answer.

For each unlabelled bullet you see the job, the premise (exactly what the
verifier sees) and the bullet, and answer one question: **does the premise,
on its own, support every claim in the bullet?** Judge the words, not the
spirit — "CI/CD" is not supported by evidence that says "CI".

To make each judgement fast, the words in the bullet that do not appear in
the evidence are listed under NEW. A bullet with nothing new is usually a
quick yes; each new word is the thing to check. The helper only points -
the judgement is yours, and a paraphrase ("cut" for "reduced") is fine.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from grounded.evals.golden import JobSet, read_items, write_items
from grounded.generation.diff import word_diff

FILLER = frozenset(
    {
        "a",
        "an",
        "and",
        "the",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "from",
        "at",
        "as",
        "into",
        "using",
        "across",
        "via",
        "its",
        "their",
        "over",
        "per",
        "that",
        "this",
        "which",
        "while",
    }
)


def new_words(premise: str, bullet: str) -> list[str]:
    """Words in the bullet the evidence does not contain (filler words ignored)."""
    added = [t for op, t in word_diff(premise, bullet) if op == "added"]
    words = [w.strip(".,;:()\"'") for chunk in added for w in chunk.split()]
    return [w for w in words if w and w.casefold() not in FILLER]


PROMPT = "[y] supported  [n] not supported  [s] skip  [q] quit > "


def label(
    path: Path,
    jobs: JobSet,
    by: str,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
    limit: int | None = None,
    paraphrases_first: bool = False,
) -> tuple[int, int]:
    """Returns (labelled now, still unlabelled).

    `paraphrases_first` asks about the bullets that add the most words their evidence lacks
    before the ones that copy it - the hard cases, where the gate is actually tested. It only
    changes the order of the questions; the file keeps its order and no answer is suggested.
    """
    items = read_items(path)
    titles = {j.id: j.title for j in jobs.jds}
    queue = list(items)
    if paraphrases_first:
        queue.sort(key=lambda i: -len(new_words(i.premise, i.text)) if i.premise else 0)
    done = 0
    for item in queue:
        if item.supported is not None:
            continue
        if limit is not None and done >= limit:
            break
        remaining = sum(i.supported is None for i in items)
        show(f"\n-- {item.id} | {titles.get(item.jd_id, item.jd_id)} | {remaining} left")
        show(f"EVIDENCE: {item.premise or '(nothing resolvable was cited)'}")
        if item.unknown_ids:
            show(f"  (also cites unknown IDs: {', '.join(item.unknown_ids)})")
        show(f"BULLET:   {item.text}")
        extra = new_words(item.premise, item.text) if item.premise else []
        listed = ", ".join(extra) if extra else "(nothing - every word is in the evidence)"
        show(f"NEW:      {listed}")
        answer = ""
        while answer not in {"y", "n", "s", "q"}:
            answer = ask(PROMPT).strip().lower()[:1]
        if answer == "q":
            break
        if answer == "s":
            continue
        item.supported = answer == "y"
        if not item.supported:
            item.note = ask("  what is unsupported? (optional) > ").strip()
        item.labelled_by, item.labelled_at = by, datetime.now(UTC)
        write_items(path, items)
        done += 1
    return done, sum(i.supported is None for i in items)
