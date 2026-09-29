"""Calibrate the entailment verifier.

    python -m grounded.verification calibrate benchmarks/verifier/pairs.yaml \\
        [--out docs/results/verifier-calibration.md] \\
        [--max-false-accept 0 --max-false-reject 0]

With limits, exits 1 if the full gate at the production threshold breaks one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from grounded.config import get_settings
from grounded.verification.calibration import breaches, markdown, rates, read_pairs, run
from grounded.verification.nli import get_verifier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="grounded.verification",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("pairs", type=Path)
    cal.add_argument("--out", type=Path)
    cal.add_argument("--max-false-accept", type=float, metavar="RATE")
    cal.add_argument("--max-false-reject", type=float, metavar="RATE")
    args = parser.parse_args(argv)

    settings = get_settings()
    verifier = get_verifier(settings.verifier, cache_dir=settings.model_cache_dir)
    pairs = read_pairs(args.pairs)
    verdicts = run(pairs, verifier)
    report = markdown(pairs, verdicts, verifier.name, settings.verifier_threshold)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    failures = breaches(
        rates(pairs, verdicts, settings.verifier_threshold),
        args.max_false_accept,
        args.max_false_reject,
    )
    for failure in failures:
        print(f"REGRESSION: full gate at {settings.verifier_threshold}: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
