from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from pathlib import Path

from sts2_training.run_log import JsonlRunEventLogger
from sts2_training.runner.episode import EpisodeResult
from sts2_training.selection_log import JsonlSelectionLogger


def test_start_new_run_run_log_always_appends_run_result(monkeypatch, tmp_path: Path) -> None:
    module = importlib.import_module("sts2_training.runner.start_new_run")
    log_path = tmp_path / "zero-selection.jsonl"
    expected = EpisodeResult(
        instance_id="instance-terminal",
        decisions_made=0,
        final_dto={"run_terminal": True, "outcome": "run_victory"},
        elapsed_s=0.125,
    )

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    monkeypatch.setattr(module, "TcpConnection", lambda **kwargs: object())
    monkeypatch.setattr(
        module,
        "AsyncTrainingApiClient",
        lambda connection, selection_logger=None: FakeClient(),
    )

    async def fake_start_new_run(client, **kwargs):
        return expected

    monkeypatch.setattr(module, "start_new_run", fake_start_new_run)
    args = argparse.Namespace(
        host="127.0.0.1",
        port=8765,
        connect_timeout=5.0,
        character_id="IRONCLAD",
        ascension=0,
        seed=1,
        decision_timeout=30.0,
        max_decisions=None,
        search_mode=None,
        beam_depth=None,
        run_log=log_path,
    )

    result = asyncio.run(module._run(args))  # noqa: SLF001
    assert result == expected
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "run_result"
    assert record["instance_id"] == "instance-terminal"
    assert record["decisions_made"] == 0
    assert record["elapsed_s"] == 0.125
    assert record["outcome"] == "run_victory"
    assert record["final_dto"] == {"run_terminal": True, "outcome": "run_victory"}
    assert isinstance(record["logged_at"], str) and record["logged_at"].endswith("Z")


def test_selection_log_cli_name_remains_a_backward_compatible_alias() -> None:
    module = importlib.import_module("sts2_training.runner.start_new_run")
    path = Path("legacy-selection-name.jsonl")
    args = module._parse_args(  # noqa: SLF001
        ["--character-id", "IRONCLAD", "--selection-log", str(path)]
    )
    assert args.run_log == path


def test_run_log_writer_keeps_selection_logger_compatibility() -> None:
    assert issubclass(JsonlRunEventLogger, JsonlSelectionLogger)
