"""Render the résumé from the evidence store - no model, every bullet a verified record.

python -m grounded.resume render --profile evidence/private/profile.yaml \\
    --out evidence/private/resume.html [--pdf evidence/private/resume.pdf] [--browser PATH]

The PDF is printed by a local Chromium-family browser in headless mode (Edge ships with
Windows; Chrome or Chromium elsewhere). Nothing is sent anywhere.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from grounded.config import get_settings
from grounded.migrate import ensure_schema
from grounded.resume.build import ResumeError, build, read_profile
from grounded.resume.html import render

_BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]
_ON_PATH = ["msedge", "google-chrome", "chromium", "chromium-browser", "chrome"]


def find_browser(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for p in _BROWSERS:
        if Path(p).is_file():
            return p
    return next((found for name in _ON_PATH if (found := shutil.which(name))), None)


def print_pdf(browser: str, html: Path, pdf: Path) -> None:
    cmd = [
        browser,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf.resolve()}",
        html.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120, env=os.environ.copy())  # noqa: S603
    if not pdf.is_file() or pdf.stat().st_size == 0:
        raise RuntimeError(f"{browser} produced no PDF")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="grounded.resume",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("render")
    r.add_argument("--profile", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--pdf", type=Path)
    r.add_argument("--browser", help="Chromium-family browser for the PDF (default: search)")
    args = ap.parse_args(argv)

    try:
        profile = read_profile(args.profile)
    except ValidationError as exc:
        print(f"{args.profile}: {exc}", file=sys.stderr)
        return 1
    engine = create_engine(get_settings().database_url)
    ensure_schema(engine)
    with Session(engine) as session:
        try:
            resume = build(session, profile)
        except ResumeError as exc:
            print(exc, file=sys.stderr)
            return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(resume), encoding="utf-8")
    print(f"{args.out}: {len(resume.cited)} cited records")
    if args.pdf:
        browser = find_browser(args.browser)
        if browser is None:
            print("no Chromium-family browser found; pass --browser", file=sys.stderr)
            return 1
        print_pdf(browser, args.out, args.pdf)
        print(f"{args.pdf}: printed with {Path(browser).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
