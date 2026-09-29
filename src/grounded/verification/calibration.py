"""How often does the gate accept an unsupported bullet?

    python -m grounded.verification calibrate benchmarks/verifier/pairs.yaml \\
        --out docs/results/verifier-calibration.md

Runs every labelled (premise, hypothesis) pair and, across a sweep of
thresholds, reports the two error rates that matter:

- **false accept** — an unsupported bullet kept. The failure ADR-001 exists
  to prevent; the threshold is chosen to push this towards zero.
- **false reject** — a supported bullet dropped. The price paid for that.

Two tables: the **full gate** (number and claim-strength checks, then NLI) —
what production actually runs — and **NLI alone**, so the contribution of
each layer is visible rather than assumed. A layer-by-layer table at the
production threshold adds one check at a time.

With `--max-false-accept` / `--max-false-reject` the command is a regression
gate: it exits 1 when the full gate at the production threshold breaks a
limit. CI runs it on every push (ADR-011).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from grounded.verification.nli import Label, Verdict, Verifier
from grounded.verification.numbers import unsupported
from grounded.verification.strength import escalations

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


class Pair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    source: str
    premise: str
    hypothesis: str
    supported: bool


class PairSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pairs: list[Pair] = Field(min_length=1)


@dataclass(frozen=True)
class Row:
    threshold: float
    false_accept: float
    false_reject: float
    accuracy: float


def read_pairs(path: Path) -> PairSet:
    return PairSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def accepted(verdict: Verdict, threshold: float) -> bool:
    return verdict.label is Label.ENTAILMENT and verdict.entailment >= threshold


def check_failure(pair: Pair, numbers: bool = True, strength: bool = True) -> str | None:
    """The deterministic checks production runs before NLI, applied to one pair."""
    if numbers:
        found = unsupported(pair.hypothesis, [pair.premise], [])
        if found:
            return "number " + ", ".join(sorted(format(n, "f") for n in found))
    if strength:
        inflated = escalations(pair.hypothesis, [pair.premise])
        if inflated:
            return "strength " + ", ".join(sorted(inflated))
    return None


def rates(
    pairs: PairSet,
    verdicts: list[Verdict],
    threshold: float,
    numbers: bool = True,
    strength: bool = True,
) -> Row:
    fa = fr = negatives = positives = 0
    for pair, verdict in zip(pairs.pairs, verdicts, strict=True):
        blocked = check_failure(pair, numbers, strength) is not None
        kept = accepted(verdict, threshold) and not blocked
        if pair.supported:
            positives += 1
            fr += not kept
        else:
            negatives += 1
            fa += kept
    return Row(
        threshold=threshold,
        false_accept=fa / negatives if negatives else 0.0,
        false_reject=fr / positives if positives else 0.0,
        accuracy=1 - (fa + fr) / len(pairs.pairs),
    )


def sweep(pairs: PairSet, verdicts: list[Verdict], full_gate: bool = False) -> list[Row]:
    return [rates(pairs, verdicts, t, numbers=full_gate, strength=full_gate) for t in THRESHOLDS]


LAYERS = (
    ("NLI alone", False, False),
    ("+ number check", True, False),
    ("+ claim-strength check (full gate)", True, True),
)


def layers(pairs: PairSet, verdicts: list[Verdict], threshold: float) -> list[tuple[str, Row]]:
    """Add one deterministic check at a time, at one threshold."""
    return [(name, rates(pairs, verdicts, threshold, n, s)) for name, n, s in LAYERS]


def breaches(row: Row, max_false_accept: float | None, max_false_reject: float | None) -> list[str]:
    failures = []
    if max_false_accept is not None and row.false_accept > max_false_accept:
        failures.append(f"false accept {row.false_accept:.0%} > {max_false_accept:.0%}")
    if max_false_reject is not None and row.false_reject > max_false_reject:
        failures.append(f"false reject {row.false_reject:.0%} > {max_false_reject:.0%}")
    return failures


def run(pairs: PairSet, verifier: Verifier) -> list[Verdict]:
    return [verifier.check(p.premise, p.hypothesis) for p in pairs.pairs]


def markdown(
    pairs: PairSet, verdicts: list[Verdict], verifier: str, threshold: float = 0.95
) -> str:
    negatives = sum(not p.supported for p in pairs.pairs)
    lines = [
        f"# Verifier calibration — {date.today().isoformat()}",
        "",
        f"Verifier `{verifier}` on {len(pairs.pairs)} labelled pairs "
        f"({negatives} unsupported, {len(pairs.pairs) - negatives} supported) from "
        "`benchmarks/verifier/pairs.yaml`. Three pairs are verbatim bullets from the first "
        "live generation; the rest are synthetic.",
    ]
    for title, full in (
        ("Full gate — checks, then NLI (what production runs)", True),
        ("NLI alone", False),
    ):
        lines += [
            "",
            f"## {title}",
            "",
            "| Threshold | False accept | False reject | Accuracy |",
            "|---|---|---|---|",
        ]
        for r in sweep(pairs, verdicts, full_gate=full):
            lines.append(
                f"| {r.threshold:.2f} | {r.false_accept:.0%} | {r.false_reject:.0%} | "
                f"{r.accuracy:.0%} |"
            )
    lines += [
        "",
        f"## Layer by layer at {threshold:.2f} (the production threshold)",
        "",
        "| Gate | False accept | False reject | Accuracy |",
        "|---|---|---|---|",
    ]
    for name, r in layers(pairs, verdicts, threshold):
        lines.append(f"| {name} | {r.false_accept:.0%} | {r.false_reject:.0%} | {r.accuracy:.0%} |")
    lines += [
        "",
        "## Per pair",
        "",
        "| Pair | Source | Supported | Checks | NLI verdict | Entailment |",
        "|---|---|---|---|---|---|",
    ]
    for p, verdict in zip(pairs.pairs, verdicts, strict=True):
        checks = check_failure(p) or "pass"
        lines.append(
            f"| {p.id} | {p.source} | {'yes' if p.supported else 'no'} | {checks} | "
            f"{verdict.label} | {verdict.entailment:.2f} |"
        )
    return "\n".join(lines) + "\n"
