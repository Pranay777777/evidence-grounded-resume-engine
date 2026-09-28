"""Manage the evidence store.

python -m grounded.evidence check  evidence/evidence.yaml   # validate only
python -m grounded.evidence load   evidence/evidence.yaml   # upsert
python -m grounded.evidence list   [--status unverified]
python -m grounded.evidence verify <id> --method artifact --by "GitHub Actions"
python -m grounded.evidence reject <id> --by "Pranay"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from grounded.config import get_settings
from grounded.evidence.enums import VerificationMethod, VerificationStatus
from grounded.evidence.models import Evidence
from grounded.evidence.schema import EvidenceFile
from grounded.evidence.store import EvidenceError, load, read_file, reject, verify
from grounded.migrate import ensure_schema


def _session() -> Session:
    engine = create_engine(get_settings().database_url)
    ensure_schema(engine)
    return Session(engine)


def _read(path: Path) -> EvidenceFile:
    try:
        return read_file(path)
    except ValidationError as exc:
        print(f"{path}: {exc.error_count()} problem(s)", file=sys.stderr)
        for err in exc.errors():
            where = ".".join(str(p) for p in err["loc"])
            print(f"  {where}: {err['msg']}", file=sys.stderr)
        raise SystemExit(1) from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="grounded.evidence",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "load"):
        cmd = sub.add_parser(name)
        cmd.add_argument("path", type=Path)
    listing = sub.add_parser("list")
    listing.add_argument("--status", choices=[s.value for s in VerificationStatus])
    ver = sub.add_parser("verify")
    ver.add_argument("id")
    ver.add_argument("--method", required=True, choices=[m.value for m in VerificationMethod])
    ver.add_argument("--by", required=True)
    rej = sub.add_parser("reject")
    rej.add_argument("id")
    rej.add_argument("--by", required=True)
    args = parser.parse_args(argv)

    if args.command == "check":
        data = _read(args.path)
        print(
            f"{args.path}: valid — {len(data.roles)} role(s), "
            f"{len(data.projects)} project(s), {len(data.evidence)} evidence record(s)"
        )
        return 0

    with _session() as session:
        try:
            if args.command == "load":
                report = load(session, _read(args.path))
                for label, ids in (("created", report.created), ("revised", report.revised)):
                    for i in ids:
                        print(f"  {label:<8} {i}")
                print(report.summary())
            elif args.command == "verify":
                r = verify(session, args.id, VerificationMethod(args.method), args.by)
                how = f"{r.verification_method}: {r.verified_by}"
                print(f"verified {r.id} (rev {r.revision}) by {how}")
            elif args.command == "reject":
                r = reject(session, args.id, args.by)
                print(f"rejected {r.id} (rev {r.revision})")
            else:
                query = select(Evidence).order_by(Evidence.id)
                if args.status:
                    query = query.where(Evidence.verification_status == args.status)
                rows = list(session.scalars(query))
                for r in rows:
                    status = f"{r.verification_status:<10}"
                    print(f"  {status} {r.id:<40} rev {r.revision}  {r.statement[:60]}")
                print(f"{len(rows)} record(s)")
        except EvidenceError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
