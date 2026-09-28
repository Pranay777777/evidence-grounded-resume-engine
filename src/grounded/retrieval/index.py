"""Keeping embeddings in step with the evidence they encode.

Only citable records are embedded — an unverified record can never be
cited, so retrieving it would only waste a slot the generator could have
used. A record is (re)embedded when it has no vector for the embedder, or
when its vector encodes an older revision: the fact changed, so the vector
describes a statement that no longer exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from grounded.evidence.enums import VerificationStatus
from grounded.evidence.models import Evidence, EvidenceEmbedding, Project
from grounded.retrieval.chunking import document_text
from grounded.retrieval.embedding import Embedder

BATCH = 64


@dataclass
class IndexReport:
    embedded: list[str] = field(default_factory=list)
    current: int = 0

    def summary(self, embedder: str) -> str:
        return f"{embedder}: embedded {len(self.embedded)}, already current {self.current}"


def citable_records(session: Session) -> list[Evidence]:
    return list(
        session.scalars(
            select(Evidence)
            .where(Evidence.verification_status == VerificationStatus.VERIFIED)
            .options(
                selectinload(Evidence.project).selectinload(Project.role),
                selectinload(Evidence.role),
            )
            .order_by(Evidence.id)
        )
    )


def embed_pending(session: Session, embedder: Embedder) -> IndexReport:
    report = IndexReport()
    existing = {
        row.evidence_id: row
        for row in session.scalars(
            select(EvidenceEmbedding).where(EvidenceEmbedding.embedder == embedder.name)
        )
    }
    pending = []
    for record in citable_records(session):
        row = existing.get(record.id)
        if row is not None and row.revision == record.revision:
            report.current += 1
        else:
            pending.append(record)

    for start in range(0, len(pending), BATCH):
        batch = pending[start : start + BATCH]
        vectors = embedder.embed([document_text(r) for r in batch])
        for record, vector in zip(batch, vectors, strict=True):
            row = existing.get(record.id) or EvidenceEmbedding(
                evidence_id=record.id, embedder=embedder.name
            )
            row.revision = record.revision
            row.embedding = vector
            session.add(row)
            report.embedded.append(record.id)
    session.commit()
    return report
