"""Golden set and evaluation.

python -m grounded.evals collect --model <pinned-id> [--out benchmarks/golden/golden.jsonl]
python -m grounded.evals label --by "Your Name" [--file benchmarks/golden/golden.jsonl]
python -m grounded.evals run [--file benchmarks/golden/golden.jsonl] [--out docs/results/eval.md]
                             [--check benchmarks/golden/limits.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from grounded.config import get_settings
from grounded.evals.collect import PinRequiredError, collect, same_model
from grounded.evals.golden import GoldenItem, merge, read_items, read_jobs, write_items
from grounded.evals.label import label
from grounded.evals.metrics import check, evaluate, markdown
from grounded.evidence.store import read_file
from grounded.generation.llm import LLMError, OpenAICompatibleClient
from grounded.retrieval.embedding import get_embedder
from grounded.verification.nli import get_verifier

GOLDEN = Path("benchmarks/golden/golden.jsonl")
JOBS = Path("benchmarks/golden/jds.yaml")
CORPUS = Path("benchmarks/retrieval/corpus.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="grounded.evals",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    col = sub.add_parser("collect")
    col.add_argument("--model", required=True, help="a pinned model id (not openrouter/free)")
    col.add_argument("--out", type=Path, default=GOLDEN)
    lab = sub.add_parser("label")
    lab.add_argument("--by", required=True)
    lab.add_argument("--file", type=Path, default=GOLDEN)
    run = sub.add_parser("run")
    run.add_argument("--file", type=Path, default=GOLDEN)
    run.add_argument("--out", type=Path)
    run.add_argument("--check", type=Path, help="YAML of metric limits; exit 1 if exceeded")
    args = parser.parse_args(argv)
    settings = get_settings()
    jobs = read_jobs(JOBS)

    if args.command == "collect":
        existing = read_items(args.out)
        skip = frozenset(i.jd_id for i in existing if same_model(i.model, args.model))

        def save(batch: list[GoldenItem]) -> None:
            merged, _ = merge(read_items(args.out), batch)
            write_items(args.out, merged)

        try:
            client = OpenAICompatibleClient(
                settings.openrouter_api_key, args.model, settings.llm_base_url
            )
            result = collect(
                CORPUS, jobs, client, get_embedder(settings.embedder), skip=skip, save=save
            )
        except (PinRequiredError, LLMError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        total = len(read_items(args.out))
        print(
            f"\n{len(result.items)} bullet(s) from {len(result.done)} job(s) saved to {args.out} "
            f"({total} in total); {len(result.skipped)} job(s) already collected for this model"
        )
        if result.failed:
            why = " after repeated provider errors" if result.stopped else ""
            print(
                f"{len(result.failed)} job(s) not collected{why}: {', '.join(result.failed)} - "
                "run the same command later (free models are rate limited upstream) or "
                "with another pinned model; finished jobs are skipped",
                file=sys.stderr,
            )
        return 0 if result.items or not result.failed else 1

    if args.command == "label":
        done, left = label(args.file, jobs, args.by, ask=input)
        print(f"\nlabelled {done} this session; {left} still unlabelled")
        return 0

    items = read_items(args.file)
    if not any(i.supported is not None for i in items):
        print(f"no labelled bullets in {args.file} - collect and label first", file=sys.stderr)
        return 1
    verifier = get_verifier(settings.verifier, cache_dir=settings.model_cache_dir)
    corpus_text = [e.statement for e in read_file(CORPUS).evidence]
    report = evaluate(items, jobs, verifier, settings.verifier_threshold, corpus_text)
    text = markdown(report, verifier.name, settings.verifier_threshold)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    if args.check:
        limits = yaml.safe_load(args.check.read_text(encoding="utf-8")) or {}
        try:
            failures = check(report, limits)
        except ValueError as exc:
            print(f"error: {args.check}: {exc}", file=sys.stderr)
            return 2
        for failure in failures:
            print(f"REGRESSION: {failure}", file=sys.stderr)
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
