"""Label golden-set bullets by hand. Resumable: progress is saved after each answer.

For each unlabelled bullet you see the job, the premise (exactly what the
verifier sees) and the bullet, and answer one question: **does the premise,
on its own, support every claim in the bullet?** Judge the words, not the
spirit — "CI/CD" is not supported by evidence that says "CI".
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from grounded.evals.golden import JobSet, read_items, write_items

PROMPT = "[y] supported  [n] not supported  [s] skip  [q] quit > "


def label(
    path: Path,
    jobs: JobSet,
    by: str,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
) -> tuple[int, int]:
    """Returns (labelled now, still unlabelled)."""
    items = read_items(path)
    titles = {j.id: j.title for j in jobs.jds}
    done = 0
    for item in items:
        if item.supported is not None:
            continue
        remaining = sum(i.supported is None for i in items)
        show(f"\n-- {item.id} | {titles.get(item.jd_id, item.jd_id)} | {remaining} left")
        show(f"EVIDENCE: {item.premise or '(nothing resolvable was cited)'}")
        if item.unknown_ids:
            show(f"  (also cites unknown IDs: {', '.join(item.unknown_ids)})")
        show(f"BULLET:   {item.text}")
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
