from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

from sts2_training.visualizer.log_reader import read_jsonl
from sts2_training.visualizer.server import VisualizerApp, make_server
from sts2_training.visualizer.store import EventStore


def _request(server, path: str) -> tuple[int, dict]:
    host, port = server.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=2)
    connection.request("GET", path)
    response = connection.getresponse()
    body = json.loads(response.read())
    connection.close()
    return response.status, body


def test_replay_mode_exposes_canonical_frames_and_run_terminal_record(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"
    records = [
        {
            "event": "selection",
            "logged_at": "2026-08-10T10:00:00Z",
            "received": {
                "decision_point_id": "d1",
                "masked_emulator_dto": {
                    "player_state": {
                        "character_id": "IRONCLAD",
                        "health": 61,
                        "max_health": 80,
                        "current_block": 5,
                        "current_energy": 2,
                        "money": 123,
                    },
                    "monsters": [
                        {"monster_id": "CULTIST", "hp": 42, "max_hp": 48, "next_move": "attack"}
                    ],
                    "cards_in_hand": [
                        {"card_id": "strike", "display_name": "Strike", "energy_cost": 1, "text": "Deal damage"}
                    ],
                    "floor_number": 7,
                    "draw_count": 3,
                    "discard_count": 4,
                    "legal_actions": [
                        {"action_id": "strike", "action_type": "play_card", "name": "Strike", "cost": 1}
                    ],
                },
            },
            "request": {"operation": "commit_action", "branch_id": "root", "action_id": "strike"},
            "selected_action_id": "strike",
            "result": {"masked_emulator_dto": {"legal_actions": []}},
        },
        {
            "event": "run_result",
            "logged_at": "2026-08-10T10:00:01Z",
            "instance_id": "instance-1",
            "decisions_made": 1,
            "elapsed_s": 0.5,
            "outcome": "run_victory",
            "final_dto": {
                "run_terminal": True,
                "outcome": "run_victory",
                "player": {"name": "IRONCLAD", "current_hp": 61, "max_hp": 80},
            },
        },
    ]
    log_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    store = EventStore(read_jsonl(log_path))
    app = VisualizerApp(mode="replay", store=store, replay_path=str(log_path))
    server = make_server(app, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status_code, payload = _request(server, "/api/events?after=-1")
        assert status_code == 200
        selection, terminal = payload["events"]

        assert selection["frame_source"] == "received.masked_emulator_dto"
        assert selection["phase"] == "selection"
        assert selection["frame"]["player"] == {
            "name": "IRONCLAD",
            "current_hp": 61,
            "max_hp": 80,
            "block": 5,
            "intent": None,
            "statuses": [],
        }
        assert selection["frame"]["enemies"][0]["name"] == "CULTIST"
        assert selection["frame"]["hand"][0]["name"] == "Strike"
        assert selection["frame"]["resources"]["gold"] == 123
        assert selection["frame"]["resources"]["floor"] == 7
        assert selection["frame"]["piles"]["draw"] == 3
        assert selection["selected_action"]["action_id"] == "strike"

        assert terminal["event"] == "run_result"
        assert terminal["frame_source"] == "final_dto"
        assert terminal["phase"] == "run_terminal"
        assert terminal["frame"]["outcome"] == "run_victory"
        assert "before" not in terminal and "after" not in terminal
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
