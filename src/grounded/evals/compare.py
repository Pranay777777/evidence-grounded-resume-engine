"""Compare models on the same job descriptions (ADR-012).

    python -m grounded.evals compare --out docs/results/model-comparison.md

For every model in the golden set: what the gate keeps, why it drops the
rest, citation precision/recall, JD keyword coverage and tone - none of which
need human labels - plus tokens, latency and cost per draft from runs.jsonl,
and the fabrication rate once bullets are labelled. Every model saw the same
job descriptions, the same retrieved evidence and a recorded prompt version.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from statistics import mean

from grounded.evals.golden import GoldenItem, JobSet, RunRecord
from grounded.evals.metrics import Report, evaluate
from grounded.verification.nli import Verifier


@dataclass(frozen=True)
class ModelRow:
    model: str
    jobs: int
    report: Report
    runs: int
    tokens_per_draft: float | None
    latency_s: float | None
    attempts: float | None

    @property
    def kept_rate(self) -> float:
        return self.report.kept / self.report.items if self.report.items else 0.0

    @property
    def cost_per_draft(self) -> float | None:
        """$0 on free tiers; unknown otherwise (prices change - not guessed)."""
        return 0.0 if self.model.endswith(":free") else None


def compare(
    items: list[GoldenItem],
    runs: list[RunRecord],
    jobs: JobSet,
    verifier: Verifier,
    threshold: float,
    corpus_text: list[str],
) -> list[ModelRow]:
    rows = []
    for model in sorted({i.model for i in items}):
        mine = [i for i in items if i.model == model]
        calls = [r for r in runs if r.model == model]
        report = evaluate(mine, jobs, verifier, threshold, corpus_text, labelled_only=False)
        rows.append(
            ModelRow(
                model=model,
                jobs=len({i.jd_id for i in mine}),
                report=report,
                runs=len(calls),
                tokens_per_draft=(
                    mean(r.prompt_tokens + r.completion_tokens for r in calls) if calls else None
                ),
                latency_s=mean(r.latency_s for r in calls) if calls else None,
                attempts=mean(r.attempts for r in calls) if calls else None,
            )
        )
    return rows


def _or_dash(value: float | None, fmt: str) -> str:
    return "-" if value is None else format(value, fmt)


def markdown(rows: list[ModelRow], verifier: str, threshold: float, total_jobs: int) -> str:
    lines = [
        f"# Model comparison - {date.today().isoformat()}",
        "",
        f"Every model answered the same {total_jobs} synthetic job descriptions "
        "(`benchmarks/golden/jds.yaml`) from the same retrieved evidence; every bullet "
        f"went through the production gate (checks + `{verifier}` >= {threshold}). "
        "Quality columns need no human labels; fabrication does.",
        "",
        "| Model | Jobs | Bullets | Kept by gate | Citation P / R | Keyword coverage "
        "| Tone | Fabrication (labelled) | Tokens / draft | Latency / draft | Cost / draft |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rep = r.report
        fabrication = (
            f"{rep.output_fabrication:.0%} of kept ({rep.labelled} labelled)"
            if rep.labelled
            else "unlabelled"
        )
        cost = r.cost_per_draft
        lines.append(
            f"| {r.model} | {r.jobs} | {rep.items} | {rep.kept} ({r.kept_rate:.0%}) | "
            f"{rep.citation_precision:.0%} / {rep.citation_recall:.0%} | "
            f"{rep.keyword_coverage:.0%} | {rep.tone:.0%} | {fabrication} | "
            f"{_or_dash(r.tokens_per_draft, ',.0f')} | "
            f"{'-' if r.latency_s is None else f'{r.latency_s:.1f} s'} | "
            f"{'-' if cost is None else f'${cost:.2f}'} |"
        )
    lines += ["", "## Why the gate dropped bullets", ""]
    for r in rows:
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(r.report.drop_reasons.items()))
        lines.append(f"- **{r.model}** - {reasons or 'nothing dropped'}")
    lines += [
        "",
        "Tokens, latency and cost come from `benchmarks/golden/runs.jsonl`; `-` means the "
        "model's bullets were collected before runs were recorded. Cost is $0 for free-tier "
        "models and not estimated for paid ones.",
    ]
    return "\n".join(lines) + "\n"
