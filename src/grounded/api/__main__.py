"""API utilities.

    python -m grounded.api keygen <name> [--budget 50000] [--tenant default]
    python -m grounded.api token --subject <who> --tenant <tenant> \\
        --scopes generate,read_evidence [--ttl-hours 1] [--budget N]

keygen prints a new API key once and the API_KEYS entry for .env - only the
key's SHA-256 digest is ever stored (ADR-016). token prints a signed JWT
(AUTH=jwt, JWT_SECRET) for local testing; production tokens would come from
an identity provider signing with the same secret (ADR-017).
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta

from grounded.api.auth import SCOPES, AuthError, issue
from grounded.api.guards import digest, new_key
from grounded.config import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="grounded.api",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("keygen", help="create an API key")
    gen.add_argument("name")
    gen.add_argument("--budget", type=int, default=50_000, help="tokens per UTC day")
    gen.add_argument("--tenant", default="default")
    tok = sub.add_parser("token", help="issue a signed JWT")
    tok.add_argument("--subject", required=True)
    tok.add_argument("--tenant", required=True)
    tok.add_argument(
        "--scopes", required=True, help=f"comma-separated: {', '.join(sorted(SCOPES))}"
    )
    tok.add_argument("--ttl-hours", type=float, default=1.0)
    tok.add_argument("--budget", type=int)
    args = parser.parse_args(argv)
    if args.command == "token":
        try:
            token = issue(
                get_settings(),
                args.subject,
                args.tenant,
                {s.strip() for s in args.scopes.split(",") if s.strip()},
                timedelta(hours=args.ttl_hours),
                args.budget,
            )
        except AuthError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(token)
        return 0
    if "=" in args.name or "," in args.name or ":" in args.name:
        parser.error("name may not contain '=', ',' or ':'")
    key = new_key()
    print(f"key (shown once, send as X-API-Key): {key}")
    entry = f"{args.name}={digest(key)}:{args.budget}"
    if args.tenant != "default":
        entry += f":{args.tenant}"
    print(f"add to API_KEYS in .env:              {entry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
