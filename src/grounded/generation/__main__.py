"""Draft résumé bullets for a job description from verified evidence.

    python -m grounded.generation draft --jd job.txt [-k 12] [--rerank] [--json]
    python -m grounded.generation models      # free models that support tool calling, live

Needs OPENROUTER_API_KEY (free at openrouter.ai/keys). The evidence that is
retrieved is sent to the configured model; on OpenRouter's free models the
provider may log prompts (ADR-007). Output is a draft: grounding checks
(steps 48 and 49) are what decide which bullets survive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, selectinload

from grounded.config import get_settings
from grounded.evidence.models import Evidence, Project
from grounded.generation.generate import GenerationError, generate
from grounded.generation.llm import FREE_ROUTER, LLMError, OpenAICompatibleClient, free_tool_models
from grounded.migrate import ensure_schema
from grounded.retrieval.embedding import get_embedder
from grounded.retrieval.rerank import get_reranker
from grounded.retrieval.search import search
from grounded.verification.grounding import ground
from grounded.verification.nli import get_verifier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="grounded.generation",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    draft = sub.add_parser("draft")
    draft.add_argument("--jd", type=Path, required=True, help="file with the job description")
    draft.add_argument("-k", type=int, default=12, help="evidence records to retrieve")
    draft.add_argument("--rerank", action="store_true")
    draft.add_argument("--json", action="store_true", help="print the draft as JSON")
    draft.add_argument(
        "--unverified",
        action="store_true",
        help="skip the entailment model (other checks still run); marks output UNVERIFIED",
    )
    sub.add_parser("models", help="list free models that support tool calling")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.command == "models":
        try:
            models = free_tool_models(settings.llm_base_url)
        except (LLMError, httpx.HTTPError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for model in models:
            print(f"  {model.id:<60} {model.context_length:>9,} ctx")
        print(
            f"{len(models)} free model(s) with tool calling. Pin one with LLM_MODEL=<id> "
            f"in .env, or keep the default router ({FREE_ROUTER})."
        )
        return 0

    job = args.jd.read_text(encoding="utf-8")
    engine = create_engine(settings.database_url)
    ensure_schema(engine)
    with Session(engine) as session:
        reranker = get_reranker("minilm") if args.rerank else None
        hits = search(session, job, get_embedder(settings.embedder), k=args.k, reranker=reranker)
        ids = [h.evidence_id for h in hits]
        loaded = session.scalars(
            select(Evidence)
            .where(Evidence.id.in_(ids))
            .options(
                selectinload(Evidence.project).selectinload(Project.role),
                selectinload(Evidence.role),
            )
        ).all()
        by_id = {r.id: r for r in loaded}
        records = [by_id[i] for i in ids]
        try:
            verifier = (
                None
                if args.unverified
                else get_verifier(settings.verifier, cache_dir=settings.model_cache_dir)
            )
            client = OpenAICompatibleClient(
                settings.openrouter_api_key, settings.llm_model, settings.llm_base_url
            )
            result = generate(client, job, records)
        except (LLMError, GenerationError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            for detail in getattr(exc, "errors", []):
                print(f"  {detail}", file=sys.stderr)
            return 1
        candidates = {r.id: r.revision for r in records}
        report = ground(session, result.draft, candidates, verifier, settings.verifier_threshold)

    if args.json:
        print(
            json.dumps(
                {
                    "model": result.model,
                    "prompt_version": result.prompt_version,
                    "attempts": result.attempts,
                    "verifier": report.verifier,
                    "threshold": report.threshold,
                    "kept": [
                        {
                            **k.bullet.model_dump(),
                            "entailment": k.entailment,
                            "citations": [c.__dict__ for c in k.citations],
                        }
                        for k in report.kept
                    ],
                    "dropped": [
                        {**d.bullet.model_dump(), "reason": d.reason} for d in report.dropped
                    ],
                },
                indent=2,
            )
        )
        return 0

    mode = f"verifier {report.verifier} ≥ {report.threshold}" if report.verified else "UNVERIFIED"
    print(
        f"model {result.model} · prompt {result.prompt_version} · attempts {result.attempts} "
        f"· {mode}\n"
    )
    for kept in report.kept:
        score = f"{kept.entailment:.2f}" if kept.entailment is not None else "not checked"
        print(f"✓ {kept.bullet.text}")
        cites = ", ".join(
            f"{c.evidence_id} (rev {c.revision}, {c.verification_method})" for c in kept.citations
        )
        print(f"  cites: {cites} · entailment {score}")
    for dropped in report.dropped:
        print(f"✗ {dropped.bullet.text}")
        print(f"  dropped: {dropped.reason}")
    print(f"\n{len(report.kept)} kept, {len(report.dropped)} dropped.")
    if not report.verified:
        print("UNVERIFIED: entailment was not checked. Do not use these bullets as they stand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
