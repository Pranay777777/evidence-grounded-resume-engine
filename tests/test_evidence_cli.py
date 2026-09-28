"""The evidence CLI, end to end against a file-backed store."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from grounded.config import get_settings
from grounded.evidence.__main__ import main

FIXTURE = Path(__file__).parent / "fixtures" / "example_evidence.yaml"


@pytest.fixture(autouse=True)
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'store.db'}")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_check_validates_without_a_database(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(FIXTURE)]) == 0
    assert "valid — 1 role(s), 1 project(s), 3 evidence record(s)" in capsys.readouterr().out


def test_check_explains_what_is_wrong(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("evidence:\n  - id: Bad_ID\n    kind: achievement\n    statement: short\n")
    with pytest.raises(SystemExit):
        main(["check", str(bad)])
    err = capsys.readouterr().err
    assert "problem(s)" in err and "evidence.0.id" in err


def test_load_then_list_then_verify(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["load", str(FIXTURE)]) == 0
    assert "3 created" in capsys.readouterr().out

    assert main(["list", "--status", "unverified"]) == 0
    assert "2 record(s)" in capsys.readouterr().out

    assert main(["verify", "example-oncall", "--method", "self_attested", "--by", "me"]) == 0
    assert "verified example-oncall (rev 1)" in capsys.readouterr().out

    assert main(["reject", "example-pipeline-built", "--by", "reviewer"]) == 0
    assert "rejected example-pipeline-built" in capsys.readouterr().out

    assert main(["list"]) == 0
    assert "3 record(s)" in capsys.readouterr().out


def test_errors_are_reported_not_raised(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verify", "missing", "--method", "self_attested", "--by", "me"]) == 1
    assert "no evidence 'missing'" in capsys.readouterr().err
