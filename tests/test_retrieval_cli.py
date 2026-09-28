"""The retrieval CLI against a file-backed store, with the hashing embedder."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from grounded.config import get_settings
from grounded.evidence.__main__ import main as evidence_main
from grounded.retrieval.__main__ import main

CORPUS = Path(__file__).parent / "fixtures" / "retrieval_corpus.yaml"


@pytest.fixture(autouse=True)
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'store.db'}")
    monkeypatch.setenv("EMBEDDER", "hashing")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_index_then_search(capsys: pytest.CaptureFixture[str]) -> None:
    evidence_main(["load", str(CORPUS)])
    assert main(["index"]) == 0
    assert "hashing: embedded 4" in capsys.readouterr().out
    assert main(["search", "HNSW vector index", "-k", "2"]) == 0
    out = capsys.readouterr().out
    assert " 1. ex-hnsw-index" in out and "bm25#1" in out


def test_bm25_mode_needs_no_embedder(capsys: pytest.CaptureFixture[str]) -> None:
    evidence_main(["load", str(CORPUS)])
    capsys.readouterr()
    assert main(["search", "pytest", "--mode", "bm25"]) == 0
    assert "ex-pytest-suite" in capsys.readouterr().out


def test_search_on_an_empty_store_says_why(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["search", "anything"]) == 0
    assert "no citable evidence" in capsys.readouterr().out
