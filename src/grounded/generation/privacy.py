"""Redact personal and confidential text before it leaves the machine (ADR-015).

Free LLM providers may log prompts (ADR-007). Before a request goes to any
provider that is not local, the job description and evidence text are
rewritten: each sensitive span becomes a placeholder such as `[EMAIL_1]`,
and the same span always gets the same placeholder within a request. The
model writes bullets with the placeholders; they are restored before the
grounding gate, which compares against the real, unredacted evidence.

Two engines:

- `patterns` (default, no dependencies): e-mail addresses, phone numbers,
  URLs, IP addresses, plus a deny-list of terms you name (`REDACT_TERMS`) -
  the place for an employer or client name.
- `presidio` (extra `[privacy]` + a spaCy model): Microsoft Presidio's
  recognisers, which also find people's names; the deny-list still applies.

Record IDs are never redacted, so citations still resolve.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "URL": re.compile(r"\b(?:https?://|www\.)[^\s<>\"']*[^\s<>\"'.,;:!?)\]]", re.IGNORECASE),
    "IP": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "PHONE": re.compile(r"(?<![\w.])\+?\d[\d ()-]{7,}\d(?![\w.])"),
}
_PLACEHOLDER = re.compile(r"\[[A-Z]+_\d+\]")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str


class Finder(Protocol):
    name: str

    def find(self, text: str) -> list[Span]: ...


def _terms(text: str, terms: Sequence[str]) -> list[Span]:
    spans = []
    for term in terms:
        if term.strip():
            for m in re.finditer(rf"(?<!\w){re.escape(term.strip())}(?!\w)", text, re.IGNORECASE):
                spans.append(Span(m.start(), m.end(), "TERM"))
    return spans


@dataclass
class PatternFinder:
    terms: Sequence[str] = ()
    name: str = "patterns"

    def find(self, text: str) -> list[Span]:
        spans = _terms(text, self.terms)
        for kind, pattern in PATTERNS.items():
            spans += [Span(m.start(), m.end(), kind) for m in pattern.finditer(text)]
        return spans


PRESIDIO_ENTITIES = ("PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "URL", "IP_ADDRESS")
"""Not ORGANIZATION or LOCATION: spaCy tags technologies (Airflow, Snowflake)
as organisations, which would blind the model to the skills it writes about.
Employer and client names belong in the deny-list instead."""


@dataclass
class PresidioFinder:
    terms: Sequence[str] = ()
    analyzer: Any = None
    min_score: float = 0.5
    name: str = "presidio"

    def __post_init__(self) -> None:
        if self.analyzer is None:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_analyzer.nlp_engine import NlpEngineProvider
            except ImportError as exc:
                raise ImportError(
                    'REDACTION=presidio needs: pip install -e ".[privacy]" && '
                    "python -m spacy download en_core_web_sm"
                ) from exc
            provider = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
                }
            )
            self.analyzer = AnalyzerEngine(
                nlp_engine=provider.create_engine(), supported_languages=["en"]
            )

    def find(self, text: str) -> list[Span]:
        results = self.analyzer.analyze(text=text, language="en", entities=list(PRESIDIO_ENTITIES))
        spans = [
            Span(r.start, r.end, str(r.entity_type).split("_")[0])
            for r in results
            if r.score >= self.min_score
        ]
        return spans + _terms(text, self.terms)


@dataclass
class Redaction:
    """One request's redactions: the same span always maps to the same placeholder."""

    finder: Finder
    forward: dict[str, str] = field(default_factory=dict)
    """casefolded original -> placeholder"""
    backward: dict[str, str] = field(default_factory=dict)
    """placeholder -> original as first seen"""
    counts: dict[str, int] = field(default_factory=dict)

    def redact(self, text: str) -> str:
        spans = sorted(self.finder.find(text), key=lambda s: (s.start, -(s.end - s.start)))
        out, cursor = [], 0
        for span in spans:
            if span.start < cursor:  # overlaps an earlier, longer span
                continue
            original = text[span.start : span.end]
            key = original.casefold()
            if key not in self.forward:
                self.counts[span.kind] = self.counts.get(span.kind, 0) + 1
                self.forward[key] = f"[{span.kind}_{self.counts[span.kind]}]"
                self.backward[self.forward[key]] = original
            out += [text[cursor : span.start], self.forward[key]]
            cursor = span.end
        return "".join(out) + text[cursor:]

    def restore(self, text: str) -> str:
        return _PLACEHOLDER.sub(lambda m: self.backward.get(m.group(0), m.group(0)), text)

    @property
    def redacted(self) -> int:
        return len(self.forward)


def get_finder(name: str, terms: Sequence[str] = ()) -> Finder | None:
    if name == "off":
        return None
    if name == "patterns":
        return PatternFinder(tuple(terms))
    if name == "presidio":
        return PresidioFinder(tuple(terms))
    raise ValueError(f"unknown redaction engine '{name}' (off, patterns, presidio)")
