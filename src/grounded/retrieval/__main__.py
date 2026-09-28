"""Embed citable evidence and search it.

python -m grounded.retrieval index  [--embedder bge-small]
python -m grounded.retrieval search "job description text" [-k 10] [--mode hybrid]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from grounded.config import get_settings
from grounded.migrate import ensure_schema
from grounded.retrieval.ablation import markdown, read_benchmark, run, select
from grounded.retrieval.embedding import EMBEDDERS, Embedder, get_embedder
from grounded.retrieval.index import embed_pending
from grounded.retrieval.rerank import get_reranker
from grounded.retrieval.search import Mode, search


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="grounded.retrieval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    idx = sub.add_parser("index")
    idx.add_argument("--embedder", choices=EMBEDDERS, default=get_settings().embedder)
    srch = sub.add_parser("search")
    srch.add_argument("query")
    srch.add_argument("-k", type=int, default=10)
    srch.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.HYBRID.value)
    srch.add_argument("--embedder", choices=EMBEDDERS, default=get_settings().embedder)
    srch.add_argument("--rerank", action="store_true", help="rerank with the cross-encoder")
    abl = sub.add_parser("ablate")
    abl.add_argument("queries", type=Path)
    abl.add_argument("--corpus", type=Path, required=True)
    abl.add_argument("--out", type=Path)
    abl.add_argument(
        "--configs",
        nargs="+",
        metavar="NAME",
        help='subset, e.g. --configs "BM25" "Hybrid / hashing"',
    )
    args = parser.parse_args(argv)

    if args.command == "ablate":
        benchmark = read_benchmark(args.queries)
        results, skipped = run(args.corpus, benchmark, select(args.configs))
        report = markdown(results, skipped, benchmark)
        print(report)
        for name, reason in skipped:
            print(f"skipped {name}: {reason}")
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(report, encoding="utf-8")
        return 0

    engine = create_engine(get_settings().database_url)
    ensure_schema(engine)
    with Session(engine) as session:
        if args.command == "index":
            indexer = get_embedder(args.embedder)
            print(embed_pending(session, indexer).summary(indexer.name))
            return 0
        mode = Mode(args.mode)
        embedder: Embedder | None = None if mode is Mode.BM25 else get_embedder(args.embedder)
        reranker = get_reranker("minilm") if args.rerank else None
        hits = search(session, args.query, embedder, k=args.k, mode=mode, reranker=reranker)
        for n, hit in enumerate(hits, start=1):
            where = " ".join(f"{name}#{rank}" for name, rank in sorted(hit.ranks.items()))
            print(f"{n:>2}. {hit.evidence_id:<36} {hit.score:.4f}  [{where}]")
            print(f"    {hit.statement}")
        if not hits:
            print("no citable evidence matched — verify records, then run 'index'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
