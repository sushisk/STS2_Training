from __future__ import annotations

import asyncio
import http.client
import json
import threading
import time
from pathlib import Path

from sts2_training.visualizer.core import EventStore
from sts2_training.visualizer.live import LiveRunConfig, LiveRunController
from sts2_training.visualizer.server import VisualizerApp, make_server


class _Writer:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[dict] = []
        self.closed = False

    def __call__(self, event) -> None:
        self.records.append(dict(event))

    def close(self) -> None:
        self.closed = True


def _request(server, method: str, path: str) -> tuple[int, dict]:
    host, port = server.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=2)
    connection.request(method, path)
    response = connection.getresponse()
    body = json.loads(response.read())
    connection.close()
    return response.status, body


def test_live_mode_start_endpoint_streams_events_while_runner_executes(tmp_path: Path) -> None:
    writer_box: list[_Writer] = []

    def writer_factory(path: Path) -> _Writer:
        writer = _Writer(path)
        writer_box.append(writer)
        return writer

    async def fake_runner(emit):
        emit({
            "event": "selection",
            "received": {"decision_point_id": "live-1", "masked_emulator_dto": {"legal_actions": [{"action_id": "strike", "action_type": "play_card"}]}},
            "request": {"operation": "commit_action", "branch_id": "root", "action_id": "strike"},
            "selected_action_id": "strike",
            "result": {"masked_emulator_dto": {"legal_actions": []}},
        })
        await asyncio.sleep(0.03)
        emit({
            "event": "selection",
            "received": {"decision_point_id": "live-2", "masked_emulator_dto": {"legal_actions": []}},
            "request": {"operation": "commit_action", "branch_id": "root", "action_id": "end_turn"},
            "selected_action_id": "end_turn",
            "result": {"masked_emulator_dto": {"run_terminal": True, "outcome": "run_victory"}},
        })
        return {"outcome": "run_victory"}

    store = EventStore()
    controller = LiveRunController(
        store=store,
        config=LiveRunConfig(),
        log_path=tmp_path / "live.jsonl",
        runner=fake_runner,
        writer_factory=writer_factory,
    )
    app = VisualizerApp(mode="live", store=store, live=controller)
    server = make_server(app, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status_code, body = _request(server, "POST", "/api/live/start")
        assert status_code == 202
        assert body["state"] == "running"

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
        assert len(writer_box[0].records) == 2
        assert writer_box[0].closed is True

        status_code, second = _request(server, "POST", "/api/live/start")
        assert status_code == 409
        assert "already" in second["error"]
    finally:
        controller.join(timeout=2)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
