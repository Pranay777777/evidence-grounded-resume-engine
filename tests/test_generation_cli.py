"""The generation CLI end to end, with the LLM replaced by a fake."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from grounded.config import get_settings
from grounded.evidence.__main__ import main as evidence_main
from grounded.generation import __main__ as cli
from grounded.generation.llm import ChatResponse, LLMError
from grounded.retrieval.__main__ import main as retrieval_main
from grounded.verification.nli import Label, Verdict

CORPUS = Path(__file__).parent / "fixtures" / "retrieval_corpus.yaml"


class FakeClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.model = "fake/model"

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        cited = "ex-delta-merge" if "ex-delta-merge" in messages[1]["content"] else "ex-hnsw-index"
        args = {
            "bullets": [
                {
                    "text": "Implemented incremental loads with Delta Lake MERGE.",
                    "evidence_ids": [cited],
                }
            ]
        }
        return ChatResponse(content=None, tool_arguments=json.dumps(args), model="fake/model")


class AlwaysEntailed:
    name = "fake-nli"

    def check(self, premise: str, hypothesis: str) -> Verdict:
        return Verdict(Label.ENTAILMENT, 0.99)


@pytest.fixture(autouse=True)
def no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_verifier", lambda name, cache_dir=None: AlwaysEntailed())


@pytest.fixture(autouse=True)
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'store.db'}")
    monkeypatch.setenv("EMBEDDER", "hashing")
    get_settings.cache_clear()
    evidence_main(["load", str(CORPUS)])
    retrieval_main(["index"])
    jd = tmp_path / "jd.txt"
    jd.write_text("Data engineer with Delta Lake incremental loads.", encoding="utf-8")
    yield jd
    get_settings.cache_clear()


def test_draft_prints_bullets_with_citations(
    store: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "OpenAICompatibleClient", FakeClient)
    capsys.readouterr()
    assert cli.main(["draft", "--jd", str(store)]) == 0
    out = capsys.readouterr().out
    assert "model fake/model · prompt generate-v1 · attempts 1 · verifier fake-nli ≥ 0.95" in out
    assert "✓ Implemented incremental loads with Delta Lake MERGE." in out
    assert "cites: ex-delta-merge (rev 1, self_attested) · entailment 0.99" in out
    assert "1 kept, 0 dropped." in out


def test_draft_as_json(
    store: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "OpenAICompatibleClient", FakeClient)
    capsys.readouterr()
    assert cli.main(["draft", "--jd", str(store), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kept"][0]["evidence_ids"] == ["ex-delta-merge"]
    assert payload["verifier"] == "fake-nli" and payload["dropped"] == []


def test_provider_errors_are_reported(
    store: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise LLMError("no API key — set OPENROUTER_API_KEY")

    monkeypatch.setattr(cli, "OpenAICompatibleClient", broken)
    assert cli.main(["draft", "--jd", str(store)]) == 1
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_models_lists_free_tool_models(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from grounded.generation.llm import ModelInfo

    monkeypatch.setattr(cli, "free_tool_models", lambda base_url: [ModelInfo("x/y:free", 32768)])
    capsys.readouterr()
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "x/y:free" in out and "1 free model(s)" in out and "openrouter/free" in out


def test_models_reports_catalogue_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def down(base_url: str) -> Any:
        raise LLMError("model catalogue returned 503")

    monkeypatch.setattr(cli, "free_tool_models", down)
    assert cli.main(["models"]) == 1
    assert "503" in capsys.readouterr().err


def test_unverified_mode_says_so(
    store: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "OpenAICompatibleClient", FakeClient)
    capsys.readouterr()
    assert cli.main(["draft", "--jd", str(store), "--unverified"]) == 0
    out = capsys.readouterr().out
    assert "· UNVERIFIED" in out and "entailment not checked" in out
    assert "Do not use these bullets as they stand" in out


def test_a_bullet_the_verifier_rejects_is_dropped_with_its_reason(
    store: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Neutral:
        name = "fake-nli"

        def check(self, premise: str, hypothesis: str) -> Verdict:
            return Verdict(Label.NEUTRAL, 0.3)

    monkeypatch.setattr(cli, "OpenAICompatibleClient", FakeClient)
    monkeypatch.setattr(cli, "get_verifier", lambda name, cache_dir=None: Neutral())
    capsys.readouterr()
    assert cli.main(["draft", "--jd", str(store)]) == 0
    out = capsys.readouterr().out
    assert "✗ Implemented incremental loads" in out
    assert "not entailed by the cited evidence (neutral, entailment 0.30 < 0.95)" in out
    assert "0 kept, 1 dropped." in out
