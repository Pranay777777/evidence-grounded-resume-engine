"""The shape every generated bullet must take.

Citation is structural (ADR-001, rule 2): a bullet without `evidence_ids`
does not parse. Whether the cited evidence actually supports the bullet is a
separate question, answered by the grounding checks (step 48) and the
entailment verifier (step 49) — never by this schema.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from grounded.evidence.models import ID_PATTERN

EvidenceId = Annotated[str, Field(pattern=ID_PATTERN, max_length=80)]


class Bullet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=15, max_length=300, description="One résumé bullet.")
    evidence_ids: list[EvidenceId] = Field(
        min_length=1,
        max_length=4,
        description="IDs of the evidence records this bullet is based on — only IDs "
        "from the evidence list provided.",
    )


class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bullets: list[Bullet] = Field(min_length=1, max_length=8)
