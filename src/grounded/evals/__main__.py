"""Golden set and evaluation.

python -m grounded.evals collect --model <pinned-id> [--prompt generate-v1]
                                 [--out benchmarks/golden/golden.jsonl]
python -m grounded.evals label --by "Your Name" [--limit 40] [--paraphrases-first]
                               [--file benchmarks/golden/golden.jsonl]
python -m grounded.evals run [--file benchmarks/golden/golden.jsonl] [--out docs/results/eval.md]
                             [--check benchmarks/golden/limits.yaml]
python -m grounded.evals compare [--out docs/results/model-comparison.md]
python -m grounded.evals redteam [--out docs/results/redteam.md] [--min-pass 1.0]
python -m grounded.evals redteam --live --model <pinned-id> [--out docs/results/redteam-live.md]

--model takes an adapter spec (ADR-012): an OpenRouter id, ollama:<model> or openai:<model>.
Each collection call is logged to runs.jsonl beside the golden set (tokens, latency).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from grounded.config import get_settings
from grounded.evals import redteam
from grounded.evals.collect import PinRequiredError, collect, same_model
from grounded.evals.compare import compare
from grounded.evals.compare import markdown as comparison_markdown
from grounded.evals.golden import (
    GoldenItem,
    append_run,
    merge,
    read_items,
    read_jobs,
    read_runs,
    write_items,
)
from grounded.evals.label import label
from grounded.evals.metrics import check, evaluate, markdown
from grounded.evidence.store import read_file
from grounded.generation.adapters import make_client, parse
from grounded.generation.llm import LLMError
from grounded.retrieval.embedding import get_embedder
from grounded.verification.nli import get_verifier

GOLDEN = Path("benchmarks/golden/golden.jsonl")
JOBS = Path("benchmarks/golden/jds.yaml")
CORPUS = Path("benchmarks/retrieval/corpus.yaml")
PAYLOADS = Path("benchmarks/redteam/payloads.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="grounded.evals",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    col = sub.add_parser("collect")
    col.add_argument("--model", required=True, help="a pinned model spec (not openrouter/free)")
    col.add_argument("--prompt", help="registered prompt version (default: PROMPT_VERSION)")
    col.add_argument("--out", type=Path, default=GOLDEN)
    lab = sub.add_parser("label")
    lab.add_argument("--by", required=True)

    lab.add_argument("--limit", type=int, help="stop after labelling this many")
    lab.add_argument(
        "--paraphrases-first",
        action="store_true",
        help="ask about bullets that add words their evidence lacks before ones that copy it",
    )
    lab.add_argument("--file", type=Path, default=GOLDEN)
    run = sub.add_parser("run")
    run.add_argument("--file", type=Path, default=GOLDEN)
    run.add_argument("--out", type=Path)
    run.add_argument("--check", type=Path, help="YAML of metric limits; exit 1 if exceeded")
    run.add_argument(
        "--skip-if-unlabelled",
        action="store_true",
        help="exit 0 (not 1) when nothing is labelled yet - for CI before labelling is done",
    )
    red = sub.add_parser("redteam", help="prompt-injection red team (OWASP LLM Top 10 2026)")
    red.add_argument("--payloads", type=Path, default=PAYLOADS)
    red.add_argument("--live", action="store_true", help="send the payloads to a real model")
    red.add_argument("--model", help="pinned model spec for --live")
    red.add_argument("--min-pass", type=float, help="exit 1 below this offline pass rate")
    red.add_argument("--out", type=Path)
    cmp = sub.add_parser("compare", help="compare the models in the golden set")
    cmp.add_argument("--file", type=Path, default=GOLDEN)
    cmp.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    settings = get_settings()
    jobs = read_jobs(JOBS)

    if args.command == "collect":
        prompt_version = args.prompt or settings.prompt_version
        existing = read_items(args.out)
        _, model_id = parse(args.model)
        skip = frozenset(
            i.jd_id
            for i in existing
            if same_model(i.model, model_id) and i.prompt_version == prompt_version
        )
        runs_path = args.out.with_name("runs.jsonl")

        def save(batch: list[GoldenItem]) -> None:
            merged, _ = merge(read_items(args.out), batch)
            write_items(args.out, merged)

        try:
            client = make_client(args.model, settings)
            result = collect(
                CORPUS,
                jobs,
                client,
                get_embedder(settings.embedder),
                skip=skip,
                save=save,
                prompt_version=prompt_version,
                log=lambda run: append_run(runs_path, run),
            )
        except (PinRequiredError, LLMError, ValueError) as exc:
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

    if args.command == "redteam":
        payloads = redteam.read_payloads(args.payloads)
        verifier = get_verifier(settings.verifier, cache_dir=settings.model_cache_dir)
        if args.live:
            if not args.model:
                print("error: --live needs --model <pinned-id>", file=sys.stderr)
                return 2
            try:
                client = make_client(args.model, settings)
            except (LLMError, ValueError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            results = redteam.run_live(
                payloads,
                CORPUS,
                client,
                get_embedder(settings.embedder),
                verifier,
                settings.verifier_threshold,
            )
            text = redteam.live_markdown(results, args.model, settings.verifier_threshold)
            if not redteam.usable(results):
                print(text)
                ran = sum(1 for r in results if not r.error)
                print(
                    f"error: only {ran} of {len(results)} payloads ran - report not written "
                    "(an earlier, complete report is kept); retry when the provider recovers",
                    file=sys.stderr,
                )
                return 1
        else:
            outcomes = redteam.run_offline(payloads, CORPUS, verifier, settings.verifier_threshold)
            text = redteam.markdown(
                outcomes, verifier.name, settings.verifier_threshold, settings.prompt_version
            )
        print(text)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8", newline="\n")
        if not args.live and args.min_pass is not None:
            rate = redteam.pass_rate(outcomes, settings.prompt_version)
            if rate < args.min_pass:
                failed = [
                    o.payload.id
                    for o in outcomes
                    if not o.passed and o.version in (None, settings.prompt_version)
                ]
                print(
                    f"REGRESSION: red-team pass rate {rate:.0%} < {args.min_pass:.0%}: "
                    f"{', '.join(failed)}",
                    file=sys.stderr,
                )
                return 1
        return 0

    if args.command == "label":
        done, left = label(
            args.file,
            jobs,
            args.by,
            ask=input,
            limit=args.limit,
            paraphrases_first=args.paraphrases_first,
        )
        print(f"\nlabelled {done} this session; {left} still unlabelled")
        return 0

    items = read_items(args.file)
    if args.command == "compare":
        if not items:
            print(f"no bullets in {args.file} - collect first", file=sys.stderr)
            return 1
        verifier = get_verifier(settings.verifier, cache_dir=settings.model_cache_dir)
        corpus_text = [e.statement for e in read_file(CORPUS).evidence]
        rows = compare(
            items,
            read_runs(args.file.with_name("runs.jsonl")),
            jobs,
            verifier,
            settings.verifier_threshold,
            corpus_text,
        )
        text = comparison_markdown(rows, verifier.name, settings.verifier_threshold, len(jobs.jds))
        print(text)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8", newline="\n")
        return 0

    if not any(i.supported is not None for i in items):
        if args.skip_if_unlabelled:
            print(f"skipped: no labelled bullets in {args.file} yet")
            return 0
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
