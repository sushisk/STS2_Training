from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

from sts2_training.visualizer.core import EventStore, read_jsonl
from sts2_training.visualizer.server import VisualizerApp, make_server


def _request(server, path: str) -> tuple[int, dict]:
    host, port = server.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=2)
    connection.request("GET", path)
    response = connection.getresponse()
    body = json.loads(response.read())
    connection.close()
    return response.status, body


def test_replay_mode_loads_log_and_supports_cursor_seek(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"
    records = [
        {
            "event": "selection",
            "logged_at": "2026-08-10T10:00:00Z",
            "received": {"decision_point_id": "d1", "masked_emulator_dto": {"legal_actions": [{"action_id": "strike", "action_type": "play_card"}]}},
            "request": {"operation": "commit_action", "branch_id": "root", "action_id": "strike"},
            "selected_action_id": "strike",
            "result": {"masked_emulator_dto": {"legal_actions": []}},
        },
        {
            "event": "selection",
            "logged_at": "2026-08-10T10:00:01Z",
            "received": {"decision_point_id": "d2", "masked_emulator_dto": {"legal_actions": [{"action_id": "defend", "action_type": "play_card"}]}},
            "request": {"operation": "emulate_actions", "branch_id": "branch-a", "action_id": "defend"},
            "selected_action_id": "defend",
            "result": {"masked_emulator_dto": {"legal_actions": []}},
        },
    ]
    log_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    store = EventStore(read_jsonl(log_path))
    app = VisualizerApp(mode="replay", store=store, replay_path=str(log_path))
    server = make_server(app, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status_code, status = _request(server, "/api/status")
        assert status_code == 200
        assert status["mode"] == "replay"
        assert status["event_count"] == 2

        status_code, payload = _request(server, "/api/events?after=0")
        assert status_code == 200
        assert [event["index"] for event in payload["events"]] == [1]
        assert payload["events"][0]["operation"] == "emulate_actions"
        assert payload["events"][0]["selected_action"]["action_id"] == "defend"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
