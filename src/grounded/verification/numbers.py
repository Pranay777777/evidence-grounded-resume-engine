"""Every number in a bullet must come from the evidence it cites.

An invented or inflated figure is the most common résumé fabrication, and the
easiest to check without a model: collect the numbers a bullet states,
collect the numbers its cited records state, and require the first set to be
inside the second. Number words up to twenty count as numbers, so "five
tables" and "5 tables" match; magnitudes are applied, so "1.6 million" is
1,600,000 whichever way it is written.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "hundred": 100,
}  # fmt: skip
_SCALE = {"thousand": 1_000, "k": 1_000, "million": 1_000_000, "m": 1_000_000,
          "billion": 1_000_000_000, "bn": 1_000_000_000}  # fmt: skip

_NUMBER = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"(?:\s*(thousand|million|billion|bn|k|m)\b)?",
    re.IGNORECASE,
)
_WORD = re.compile(r"\b(" + "|".join(_WORDS) + r")\b", re.IGNORECASE)


def numbers(text: str) -> set[Decimal]:
    """Every quantity stated in `text`, normalised to a Decimal."""
    found: set[Decimal] = set()
    for match in _NUMBER.finditer(text):
        raw, scale = match.group(1).replace(",", ""), match.group(2)
        try:
            value = Decimal(raw)
        except InvalidOperation:  # pragma: no cover — the pattern only matches digits
            continue
        found.add(value)
        if scale:
            found.add(value * _SCALE[scale.lower()])
    for match in _WORD.finditer(text):
        found.add(Decimal(_WORDS[match.group(1).lower()]))
    return {n.normalize() for n in found}


def unsupported(
    bullet: str, evidence_texts: list[str], metric_values: list[Decimal | None]
) -> set[Decimal]:
    """Numbers in the bullet that no cited record states."""
    supported: set[Decimal] = set()
    for text in evidence_texts:
        supported |= numbers(text)
    supported |= {v.normalize() for v in metric_values if v is not None}
    return numbers(bullet) - supported
