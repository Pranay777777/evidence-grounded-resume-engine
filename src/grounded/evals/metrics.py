"""Eval metrics over the labelled golden set (step 51).

Headline: **output fabrication rate** — of the bullets the gate lets through,
the share a human judged unsupported. That is what a user would actually
see. Around it:

- **raw fabrication rate** — the same, before the gate: what the model writes.
- **gate false accept / false reject** — the gate's errors on real generations.
- **citation precision / recall** — kept bullets' citations against the
  records each job description is labelled as needing.
- **JD keyword coverage** — the share of *attainable* job terms (ones some
  evidence mentions) that kept bullets use. Terms no evidence supports are
  excluded: leaving them out is correct behaviour, not a coverage gap.
- **tone consistency** — the share of kept bullets meeting simple style rules.
  Crude by design and reported as such.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from grounded.evals.golden import GoldenItem, JobSet
from grounded.retrieval.embedding import tokens
from grounded.retrieval.search import STOPWORDS
from grounded.verification.nli import Label, Verdict, Verifier
from grounded.verification.numbers import unsupported
from grounded.verification.strength import escalations

FIRST_PERSON = re.compile(r"\b(i|me|my|mine|we|our|ours)\b", re.IGNORECASE)
WEAK_OPENERS = frozenset({"the", "a", "an", "this", "these", "that", "there", "it"})


@dataclass(frozen=True)
class Decision:
    item: GoldenItem
    kept: bool
    reason: str


def gate(item: GoldenItem, verifier: Verifier, threshold: float) -> Decision:
    """The production gate, applied to a stored bullet and its stored premise."""
    if item.unknown_ids:
        return Decision(item, False, "unknown citation")
    if not item.premise:
        return Decision(item, False, "nothing cited")
    if unsupported(item.text, [item.premise], []):
        return Decision(item, False, "number")
    if escalations(item.text, [item.premise]):
        return Decision(item, False, "claim strength")
    verdict: Verdict = verifier.check(item.premise, item.text)
    if verdict.label is Label.ENTAILMENT and verdict.entailment >= threshold:
        return Decision(item, True, f"entailed {verdict.entailment:.2f}")
    return Decision(item, False, f"{verdict.label} {verdict.entailment:.2f}")


def tone_ok(text: str) -> bool:
    words = text.split()
    return (
        not FIRST_PERSON.search(text)
        and bool(words)
        and words[0][:1].isupper()
        and words[0].lower().strip(",.") not in WEAK_OPENERS
        and text.count(". ") == 0  # one sentence
    )


def _terms(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in STOPWORDS and len(t) > 2}


@dataclass
class Report:
    items: int = 0
    labelled: int = 0
    kept: int = 0
    raw_fabrication: float = 0.0
    output_fabrication: float = 0.0
    false_accept: float = 0.0
    false_reject: float = 0.0
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    keyword_coverage: float = 0.0
    tone: float = 0.0
    models: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    """`version (fingerprint)` of every prompt behind the scored bullets."""
    drop_reasons: dict[str, int] = field(default_factory=dict)
    false_accepts: list[str] = field(default_factory=list)
    """IDs of unsupported bullets the gate kept — the list to read first."""


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(
    items: list[GoldenItem],
    jobs: JobSet,
    verifier: Verifier,
    threshold: float,
    corpus_text: list[str],
    labelled_only: bool = True,
) -> Report:
    """Score the gate on stored bullets.

    By default only labelled bullets count. `labelled_only=False` gates every
    bullet - what model comparison needs - while the label-based rates still
    use only the labelled ones.
    """
    labelled = [i for i in items if i.supported is not None]
    population = labelled if labelled_only else items
    decisions = [gate(i, verifier, threshold) for i in population]
    kept = [d for d in decisions if d.kept]
    unsupported_all = [d for d in decisions if d.item.supported is False]
    supported_all = [d for d in decisions if d.item.supported is True]
    kept_labelled = [d for d in kept if d.item.supported is not None]

    report = Report(
        items=len(items),
        labelled=len(labelled),
        kept=len(kept),
        models=sorted({i.model for i in items}),
        prompts=sorted(
            {
                f"{i.prompt_version} ({i.prompt_fingerprint})"
                if i.prompt_fingerprint
                else i.prompt_version
                for i in items
            }
        ),
    )
    report.raw_fabrication = _ratio(len(unsupported_all), len(labelled))
    report.output_fabrication = _ratio(
        sum(d.item.supported is False for d in kept_labelled), len(kept_labelled)
    )
    report.false_accept = _ratio(sum(d.kept for d in unsupported_all), len(unsupported_all))
    report.false_reject = _ratio(sum(not d.kept for d in supported_all), len(supported_all))
    report.false_accepts = [d.item.id for d in unsupported_all if d.kept]
    for d in decisions:
        if not d.kept:
            fixed = {"unknown citation", "nothing cited", "number", "claim strength"}
            key = d.reason if d.reason in fixed else d.reason.split(" ")[0]
            report.drop_reasons[key] = report.drop_reasons.get(key, 0) + 1

    vocabulary: set[str] = set()
    for text in corpus_text:
        vocabulary |= _terms(text)
    precision, recall, coverage = [], [], []
    for job in jobs.jds:
        job_kept = [d.item for d in kept if d.item.jd_id == job.id]
        if not any(i.jd_id == job.id for i in population):
            continue
        cited = {e for i in job_kept for e in i.evidence_ids}
        relevant = set(job.relevant)
        if cited:
            precision.append(len(cited & relevant) / len(cited))
        if relevant:
            recall.append(len(cited & relevant) / len(relevant))
        attainable = _terms(job.text) & vocabulary
        if attainable:
            used = set().union(*(_terms(i.text) for i in job_kept)) if job_kept else set()
            coverage.append(len(attainable & used) / len(attainable))
    report.citation_precision = sum(precision) / len(precision) if precision else 0.0
    report.citation_recall = sum(recall) / len(recall) if recall else 0.0
    report.keyword_coverage = sum(coverage) / len(coverage) if coverage else 0.0
    report.tone = _ratio(sum(tone_ok(d.item.text) for d in kept), len(kept))
    return report


def check(report: Report, limits: dict[str, dict[str, float]]) -> list[str]:
    """Metric limits (`max:` / `min:` maps) the report breaks. Unknown names are errors."""
    failures: list[str] = []
    for bound in limits:
        if bound not in {"max", "min"}:
            raise ValueError(f"limits must be under 'max' or 'min', not '{bound}'")
    for bound, entries in limits.items():
        for name, limit in (entries or {}).items():
            value = getattr(report, name, None)
            if not isinstance(value, float):
                raise ValueError(f"'{name}' is not a numeric metric")
            if bound == "max" and value > limit:
                failures.append(f"{name} {value:.2f} > {limit}")
            if bound == "min" and value < limit:
                failures.append(f"{name} {value:.2f} < {limit}")
    return failures


def _pct(value: float) -> str:
    return f"{value:.0%}"


def markdown(report: Report, verifier: str, threshold: float) -> str:
    lines = [
        "# Evaluation - golden set",
        "",
        f"{report.labelled} human-labelled bullets (of {report.items}) generated by "
        f"{', '.join(report.models) or '-'} with prompt {', '.join(report.prompts) or '-'}; "
        f"gate = checks + `{verifier}` >= {threshold}.",
        "",
        "| Metric | Value |",
        "|---|---|",
        "| **Output fabrication rate** (unsupported among kept) | "
        f"**{_pct(report.output_fabrication)}** |",
        f"| Raw fabrication rate (before the gate) | {_pct(report.raw_fabrication)} |",
        f"| Gate false accept | {_pct(report.false_accept)} |",
        f"| Gate false reject | {_pct(report.false_reject)} |",
        f"| Bullets kept | {report.kept} of {report.labelled} |",
        f"| Citation precision / recall | {_pct(report.citation_precision)} / "
        f"{_pct(report.citation_recall)} |",
        f"| JD keyword coverage (attainable terms) | {_pct(report.keyword_coverage)} |",
        f"| Tone consistency (rule-based) | {_pct(report.tone)} |",
        "",
        "Drop reasons: "
        + (", ".join(f"{k} {v}" for k, v in sorted(report.drop_reasons.items())) or "none"),
        "",
        "False accepts (read these first): " + (", ".join(report.false_accepts) or "none"),
    ]
    return "\n".join(lines) + "\n"
