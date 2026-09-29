"""API utilities.

    python -m grounded.api keygen <name> [--budget 50000]

Prints a new API key once, and the API_KEYS entry to add to .env. The key
itself is never stored - only its SHA-256 digest (ADR-016).
"""

from __future__ import annotations

import argparse

from grounded.api.guards import digest, new_key


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
    args = parser.parse_args(argv)
    if "=" in args.name or "," in args.name or ":" in args.name:
        parser.error("name may not contain '=', ',' or ':'")
    key = new_key()
    print(f"key (shown once, send as X-API-Key): {key}")
    print(f"add to API_KEYS in .env:              {args.name}={digest(key)}:{args.budget}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
