"""A semantic cache for drafts (ADR-013).

A draft is reused only when all of these match the earlier request:

- the model spec and the prompt fingerprint (ADR-012) - a new prompt or
  model is a new question;
- the **exact evidence set**, by record ID and revision - a draft's citations
  are only meaningful against the records it was written from, and an edited
  record is a new revision (ADR-002);
- the job description, exactly (after whitespace and case folding) or by
  embedding cosine similarity at or above the threshold.

A cached draft still goes through the full grounding gate on every use, so a
hit can save a model call but can never let an unverified bullet through.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from grounded.generation.schema import Draft
from grounded.retrieval.embedding import Embedder


def normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


def request_key(model: str, prompt_fingerprint: str, evidence: Sequence[tuple[str, int]]) -> str:
    """Everything that must match exactly, hashed. Evidence order does not matter."""
    records = ",".join(f"{i}@{rev}" for i, rev in sorted(evidence))
    return hashlib.sha256(f"{model}\n{prompt_fingerprint}\n{records}".encode()).hexdigest()[:16]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class Entry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str
    jd_hash: str
    vector: list[float]
    draft: Draft
    model: str
    prompt_version: str
    tokens: int = 0
    """Tokens the original generation used - what each hit saves."""
    hits: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class Hit:
    entry: Entry
    similarity: float


@dataclass(frozen=True)
class Stats:
    entries: int
    hits: int
    tokens_saved: int

    @property
    def hit_rate(self) -> float:
        """Hits over lookups; every stored entry began as a miss."""
        lookups = self.hits + self.entries
        return self.hits / lookups if lookups else 0.0


class SemanticCache:
    def __init__(self, path: Path, embedder: Embedder, threshold: float) -> None:
        self.path = path
        self.embedder = embedder
        self.threshold = threshold
        self.entries = self._read()

    def _read(self) -> list[Entry]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [Entry.model_validate_json(line) for line in lines if line.strip()]

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as out:
                for entry in self.entries:
                    out.write(json.dumps(entry.model_dump(mode="json")) + "\n")
            Path(tmp).replace(self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def lookup(self, key: str, job_description: str) -> Hit | None:
        candidates = [e for e in self.entries if e.key == key]
        if not candidates:
            return None
        digest = hashlib.sha256(normalise(job_description).encode()).hexdigest()
        best: Hit | None = next((Hit(e, 1.0) for e in candidates if e.jd_hash == digest), None)
        if best is None:
            vector = self.embedder.embed([job_description])[0]
            scored = [Hit(e, cosine(vector, e.vector)) for e in candidates]
            top = max(scored, key=lambda h: h.similarity)
            best = top if top.similarity >= self.threshold else None
        if best is not None:
            best.entry.hits += 1
            self._write()
        return best

    def store(
        self,
        key: str,
        job_description: str,
        draft: Draft,
        model: str,
        prompt_version: str,
        tokens: int,
    ) -> Entry:
        entry = Entry(
            key=key,
            jd_hash=hashlib.sha256(normalise(job_description).encode()).hexdigest(),
            vector=self.embedder.embed([job_description])[0],
            draft=draft,
            model=model,
            prompt_version=prompt_version,
            tokens=tokens,
        )
        self.entries.append(entry)
        self._write()
        return entry

    def stats(self) -> Stats:
        return Stats(
            entries=len(self.entries),
            hits=sum(e.hits for e in self.entries),
            tokens_saved=sum(e.hits * e.tokens for e in self.entries),
        )
