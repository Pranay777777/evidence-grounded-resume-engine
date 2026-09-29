"""Decide which bullets survive. Drop the rest, with reasons. Rewrite nothing.

Checks run cheapest first, and the first failure rejects the bullet:

1. **Citations resolve** — every cited ID was in the evidence handed to the
   model (a model can invent an ID as easily as a fact).
2. **Citations are current** — each record is still verified, at the
   revision that was retrieved. A record edited mid-request cannot lend its
   old verification to a new claim (ADR-002).
3. **Numbers are supported** — every quantity in the bullet appears in the
   records it cites.
4. **Claim strength is supported** — ownership and leadership words ("led",
   "owned", "architected"…) appear in a cited statement. The NLI model
   cannot see these upgrades; calibration proved it.
5. **The evidence entails the bullet** — an NLI model reads the cited
   statements as premise and the bullet as hypothesis. Only entailment above
   the threshold passes; neutral and contradiction both fail (ADR-009).

The premise is the cited *statements*, plus the name of each cited record's
project: the project link is part of the verified fact (it is in the content
hash), so "…for the metadata-driven-lakehouse project" is supported. Project
summaries and skills tags are not — they help retrieval, but they are not
verified, so they cannot be what makes a claim true.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from grounded.evidence.models import Evidence
from grounded.generation.schema import Bullet, Draft
from grounded.verification.markup import markup
from grounded.verification.nli import Label, Verifier
from grounded.verification.numbers import unsupported
from grounded.verification.strength import escalations

DEFAULT_THRESHOLD = 0.95
"""Entailment probability a bullet needs. Set from the first calibration run
(2026-09-28): raising it from 0.8 to 0.95 halved false accepts (31% → 15%)
and rejected no supported pair. See docs/results/verifier-calibration.md."""


@dataclass(frozen=True)
class Citation:
    evidence_id: str
    revision: int
    verification_method: str | None


@dataclass(frozen=True)
class Kept:
    bullet: Bullet
    citations: list[Citation]
    entailment: float | None
    """None only when the verifier was deliberately skipped."""


@dataclass(frozen=True)
class Dropped:
    bullet: Bullet
    reason: str


@dataclass
class GroundingReport:
    kept: list[Kept] = field(default_factory=list)
    dropped: list[Dropped] = field(default_factory=list)
    verifier: str | None = None
    threshold: float = DEFAULT_THRESHOLD

    @property
    def verified(self) -> bool:
        """False when entailment was not checked — the output must say so."""
        return self.verifier is not None


def _citation_problem(
    session: Session, bullet: Bullet, candidates: dict[str, int]
) -> tuple[str | None, list[Evidence]]:
    records = []
    for evidence_id in bullet.evidence_ids:
        if evidence_id not in candidates:
            return f"cites '{evidence_id}', which was not in the evidence provided", []
        record = session.get(Evidence, evidence_id)
        if record is None or not record.citable:
            return f"cites '{evidence_id}', which is no longer verified", []
        if record.revision != candidates[evidence_id]:
            return (
                f"cites '{evidence_id}', which changed since retrieval "
                f"(rev {candidates[evidence_id]} → {record.revision})",
                [],
            )
        records.append(record)
    return None, records


def premise(records: list[Evidence]) -> str:
    """Cited statements, then the projects they belong to — nothing unverified."""
    parts = [r.statement.strip() for r in records]
    for name in dict.fromkeys(r.project.name for r in records if r.project is not None):
        parts.append(f"This was part of the {name} project.")
    return " ".join(parts)


def ground(
    session: Session,
    draft: Draft,
    candidates: dict[str, int],
    verifier: Verifier | None,
    threshold: float = DEFAULT_THRESHOLD,
) -> GroundingReport:
    """Check every bullet. `candidates` maps each evidence ID the model was
    given to the revision it was given at."""
    report = GroundingReport(verifier=verifier.name if verifier else None, threshold=threshold)
    for bullet in draft.bullets:
        problem, records = _citation_problem(session, bullet, candidates)
        if problem:
            report.dropped.append(Dropped(bullet, problem))
            continue

        facts = [r.statement for r in records] + [r.month for r in records if r.month]
        formatting = markup(bullet.text, facts)
        if formatting:
            report.dropped.append(
                Dropped(bullet, f"contains {', '.join(formatting)} - bullets are plain text")
            )
            continue

        extra = unsupported(bullet.text, facts, [r.metric_value for r in records])
        if extra:
            stated = ", ".join(sorted(format(n, "f") for n in extra))
            report.dropped.append(Dropped(bullet, f"states {stated}, not in the cited evidence"))
            continue

        inflated = escalations(bullet.text, [r.statement for r in records])
        if inflated:
            words = ", ".join(f"'{w}'" for w in sorted(inflated))
            report.dropped.append(
                Dropped(bullet, f"claims {words}, which the cited evidence does not state")
            )
            continue

        citations = [Citation(r.id, r.revision, r.verification_method) for r in records]
        if verifier is None:
            report.kept.append(Kept(bullet, citations, None))
            continue
        verdict = verifier.check(premise(records), bullet.text)
        score = f"entailment {verdict.entailment:.2f}"
        if verdict.label is Label.CONTRADICTION:
            report.dropped.append(Dropped(bullet, f"contradicted by the cited evidence ({score})"))
        elif verdict.label is not Label.ENTAILMENT or verdict.entailment < threshold:
            reason = f"not entailed by the cited evidence ({verdict.label}, {score} < {threshold})"
            report.dropped.append(Dropped(bullet, reason))
        else:
            report.kept.append(Kept(bullet, citations, verdict.entailment))
    return report
