"""Hugging Face Space entry point (Gradio SDK, free CPU tier) - ADR-020.

The Space installs the tagged release from ``requirements.txt``. This script
fetches the demo data at the same tag, then starts the project's own server
on port 7860. No Gradio UI is used; the SDK is only the free runtime.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

REF = "v1.0.1"
REPO = "https://raw.githubusercontent.com/Pranay777777/evidence-grounded-resume-engine"
DATA = (
    "benchmarks/retrieval/corpus.yaml",
    "benchmarks/golden/golden.jsonl",
    "benchmarks/golden/jds.yaml",
)
ROOT = Path("/tmp/grounded-demo")  # noqa: S108
DEFAULTS = {
    "DEMO": "true",
    "AUTH": "off",
    "HOST": "0.0.0.0",  # noqa: S104
    "PORT": "7860",
    "EMBEDDER": "bge-small",
    "DATABASE_URL": f"sqlite:///{ROOT}/demo.db",
    "MODEL_CACHE_DIR": f"{ROOT}/models",
    "SEMANTIC_CACHE_PATH": f"{ROOT}/semantic-cache.jsonl",
    "HF_HOME": f"{ROOT}/huggingface",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def fetch() -> None:
    """Download the synthetic career and golden set at the tagged release."""
    for name in DATA:
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            url = f"{REPO}/{REF}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
                target.write_bytes(response.read())


if __name__ == "__main__":
    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)
    fetch()
    os.chdir(ROOT)  # the demo reads benchmarks/... relative to the working directory

    from grounded.demo import main

    raise SystemExit(main([]))
