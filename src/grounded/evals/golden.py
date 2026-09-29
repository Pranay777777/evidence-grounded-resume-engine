"""Golden-set records: generated bullets with a human judgement each.

Stored as JSON Lines — one bullet per line, diff-friendly, appendable. Each
record carries everything needed to judge it later without the database: the
job it was written for, the model and prompt that wrote it, the exact premise
(cited statements plus project names) at the time, and the human label.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    title: str
    text: str
    relevant: list[str] = Field(default_factory=list)


class JobSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    jds: list[Job] = Field(min_length=1)


class GoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    jd_id: str
    model: str
    prompt_version: str
    text: str
    evidence_ids: list[str]
    premise: str
    unknown_ids: list[str] = Field(default_factory=list)
    """Cited IDs that were not in the evidence given to the model."""
    supported: bool | None = None
    """The human judgement: does the premise support the bullet? None = unlabelled."""
    note: str = ""
    labelled_by: str | None = None
    labelled_at: datetime | None = None
    prompt_fingerprint: str = ""
    """The registered prompt's hash (ADR-012); empty for items collected before it."""


class RunRecord(BaseModel):
    """One generation call during collection: what it cost, not what it said."""

    model_config = ConfigDict(extra="forbid")
    jd_id: str
    model: str
    requested: str
    """The spec asked for; `model` is what the provider reports it ran."""
    prompt_version: str
    prompt_fingerprint: str
    attempts: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float
    bullets: int
    at: datetime


def read_runs(path: Path) -> list[RunRecord]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [RunRecord.model_validate_json(line) for line in lines if line.strip()]


def append_run(path: Path, run: RunRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as out:
        out.write(run.model_dump_json() + "\n")


def item_id(jd_id: str, model: str, text: str) -> str:
    """Content-addressed: the same bullet from the same model always gets the same ID.

    Positional IDs (`jd01-1`) collide when collect runs again or with a
    second model, and a collision would overwrite or hide a labelled bullet.
    """
    digest = hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()[:10]
    return f"{jd_id}-{digest}"


def merge(existing: list[GoldenItem], new: list[GoldenItem]) -> tuple[list[GoldenItem], int]:
    """Append new bullets; never drop or overwrite an existing one (labelled or not).

    Returns the merged list and how many items were added.
    """
    seen = {i.id for i in existing}
    added: list[GoldenItem] = []
    for item in new:
        if item.id not in seen:
            seen.add(item.id)
            added.append(item)
    return [*existing, *added], len(added)


def read_jobs(path: Path) -> JobSet:
    return JobSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def read_items(path: Path) -> list[GoldenItem]:
    if not path.exists():
        return []
    return [
        GoldenItem.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_items(path: Path, items: list[GoldenItem]) -> None:
    """Atomic: a crash mid-write never leaves half a golden set."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as out:
            for item in items:
                out.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n")
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
