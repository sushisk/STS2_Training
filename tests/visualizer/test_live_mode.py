from __future__ import annotations

import http.client
import json
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

from sts2_training.visualizer.live import LiveRunController
from sts2_training.visualizer.server import VisualizerApp, make_server
from sts2_training.visualizer.store import EventStore


class _FakeProcess:
    def __init__(self, done: threading.Event) -> None:
        self._done = done

    def poll(self) -> int | None:
        return 0 if self._done.is_set() else None


def _request(server, method: str, path: str) -> tuple[int, dict]:
    host, port = server.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=2)
    connection.request(method, path)
    response = connection.getresponse()
    body = json.loads(response.read())
    connection.close()
    return response.status, body


def test_live_mode_starts_runner_cli_tails_complete_records_and_can_restart(tmp_path: Path) -> None:
    log_path = tmp_path / "live.jsonl"
    commands: list[list[str]] = []
    partial_written = threading.Event()
    allow_finish = threading.Event()
    done = threading.Event()

    records = [
        {
            "event": "selection",
            "received": {
                "decision_point_id": "live-1",
                "masked_emulator_dto": {
                    "legal_actions": [{"action_id": "strike", "action_type": "play_card"}]
                },
            },
            "request": {
                "operation": "commit_action",
                "branch_id": "root",
                "action_id": "strike",
            },
            "selected_action_id": "strike",
            "result": {"masked_emulator_dto": {"legal_actions": []}},
        },
        {
            "event": "selection",
            "received": {
                "decision_point_id": "live-2",
                "masked_emulator_dto": {"legal_actions": []},
            },
            "request": {
                "operation": "commit_action",
                "branch_id": "root",
                "action_id": "end_turn",
            },
            "selected_action_id": "end_turn",
            "result": {
                "masked_emulator_dto": {"run_terminal": True, "outcome": "run_victory"}
            },
        },
    ]

    def process_factory(command: Sequence[str]) -> _FakeProcess:
        commands.append(list(command))
        run_log_index = command.index("--run-log")
        assert Path(command[run_log_index + 1]) == log_path

        first = json.dumps(records[0], separators=(",", ":")) + "\n"
        second = json.dumps(records[1], separators=(",", ":"))
        split_at = len(second) // 2

        def writer() -> None:
            with log_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(first)
                stream.write(second[:split_at])
                stream.flush()
                partial_written.set()
                allow_finish.wait(timeout=2)
                stream.write(second[split_at:] + "\n")
                stream.flush()
            done.set()

        threading.Thread(target=writer, daemon=True).start()
        return _FakeProcess(done)

    store = EventStore()
    controller = LiveRunController(
        store=store,
        log_path=log_path,
        runner_args=["--", "--host", "127.0.0.1", "--character-id", "IRONCLAD"],
        process_factory=process_factory,
    )
    app = VisualizerApp(mode="live", store=store, live=controller)
    server = make_server(app, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status_code, body = _request(server, "POST", "/api/live/start")
        assert status_code == 202
        assert body["state"] == "running"
        assert partial_written.wait(timeout=2)

        status_code, payload = _request(server, "GET", "/api/events?after=-1")
        assert status_code == 200
        assert [event["decision_point_id"] for event in payload["events"]] == ["live-1"]

        command = commands[0]
        assert command[:3] == [sys.executable, "-m", "sts2_training.runner.start_new_run"]
        assert command[3:7] == ["--host", "127.0.0.1", "--character-id", "IRONCLAD"]
        assert command.count("--run-log") == 1
        assert "--selection-log" not in command

        allow_finish.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            _, status = _request(server, "GET", "/api/status")
            if status["state"] == "completed":
                break
            time.sleep(0.01)
        else:
            raise AssertionError("live runner did not complete")

        _, payload = _request(server, "GET", "/api/events?after=-1")
        assert [event["decision_point_id"] for event in payload["events"]] == ["live-1", "live-2"]
        assert payload["events"][0]["selected_action_id"] == "strike"

        # A completed run can be started again on the same visualizer instance. The
        # controller must reset both the JSONL reader offset and the in-memory store.
        done.clear()
        partial_written.clear()
        status_code, second = _request(server, "POST", "/api/live/start")
        assert status_code == 202
        assert second["state"] in {"running", "completed"}
        assert len(commands) == 2
        assert partial_written.wait(timeout=2)

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            _, status = _request(server, "GET", "/api/status")
            if status["state"] == "completed":
                break
            time.sleep(0.01)
        else:
            raise AssertionError("restarted live runner did not complete")

        _, restarted_payload = _request(server, "GET", "/api/events?after=-1")
        assert [event["decision_point_id"] for event in restarted_payload["events"]] == [
            "live-1",
            "live-2",
        ]
    finally:
        allow_finish.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_live_mode_renders_zero_selection_run_result(tmp_path: Path) -> None:
    log_path = tmp_path / "terminal-only.jsonl"
    done = threading.Event()
    terminal = {
        "event": "run_result",
        "instance_id": "already-terminal",
        "decisions_made": 0,
        "elapsed_s": 0.01,
        "outcome": "run_victory",
        "final_dto": {
            "run_terminal": True,
            "outcome": "run_victory",
            "player": {"name": "IRONCLAD", "current_hp": 80, "max_hp": 80},
        },
    }

    def process_factory(command: Sequence[str]) -> _FakeProcess:
        with log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(terminal, separators=(",", ":")) + "\n")
            stream.flush()
        done.set()
        return _FakeProcess(done)

    store = EventStore()
    controller = LiveRunController(
        store=store,
        log_path=log_path,
        runner_args=["--", "--character-id", "IRONCLAD"],
        process_factory=process_factory,
    )
    app = VisualizerApp(mode="live", store=store, live=controller)
    server = make_server(app, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status_code, _ = _request(server, "POST", "/api/live/start")
        assert status_code == 202
        _, status = _request(server, "GET", "/api/status")
        assert status["state"] == "completed"

        _, payload = _request(server, "GET", "/api/events?after=-1")
        assert len(payload["events"]) == 1
        event = payload["events"][0]
        assert event["event"] == "run_result"
        assert event["phase"] == "run_terminal"
        assert event["frame_source"] == "final_dto"
        assert event["frame"]["outcome"] == "run_victory"
        assert event["frame"]["player"]["current_hp"] == 80
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
