from __future__ import annotations

import json
from datetime import datetime, timezone

from sts2_training.selection_log import JsonlSelectionLogger, SelectionAudit


def _decision(
    *,
    decision_point_id: str,
    boundary: str,
    room_context: dict,
    **dto_fields: object,
) -> dict:
    return {
        "status": "completed",
        "instance_id": "inst-001",
        "branch_id": "root",
        "decision_point_id": decision_point_id,
        "masked_emulator_dto": {
            "boundary": boundary,
            "room_context": room_context,
            "legal_actions": [],
            **dto_fields,
        },
    }


def _commit_request(*, request_id: str = "session-a:2") -> dict:
    return {
        "schema_version": "0.7",
        "client_session_id": "session-a",
        "request_seq": 2,
        "request_id": request_id,
        "operation": "commit_action",
        "instance_id": "inst-001",
        "branch_id": "root",
        "rng_id": 0,
        "decision_point_id": "decision-1",
        "action_id": "action-1",
    }


def test_jsonl_selection_logger_writes_utf8_and_timestamp(tmp_path) -> None:
    path = tmp_path / "logs" / "selection.jsonl"
    with JsonlSelectionLogger(
        path,
        append=False,
        clock=lambda: datetime(2026, 8, 7, 1, 2, 3, tzinfo=timezone.utc),
    ) as logger:
        logger({"event": "selection", "label": "防御"})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "event": "selection",
        "label": "防御",
        "logged_at": "2026-08-07T01:02:03Z",
    }


def test_root_commit_records_room_result() -> None:
    events: list[dict] = []
    audit = SelectionAudit(events.append)
    audit.remember(
        _decision(
            decision_point_id="decision-1",
            boundary="combat",
            room_context={"room_id": 7},
        )
    )
    result = _decision(
        decision_point_id="decision-2",
        boundary="map_select",
        room_context={"completed_room_id": 7, "outcome": "victory"},
    )

    audit.record_action(
        _commit_request(),
        source_branch_id="root",
        result=result,
    )

    assert len(events) == 1
    assert events[0]["event"] == "selection"
    assert events[0]["received"]["decision_point_id"] == "decision-1"
    assert events[0]["room_result"]["room_context_after"]["outcome"] == "victory"


def test_false_run_outcome_is_preserved() -> None:
    events: list[dict] = []
    audit = SelectionAudit(events.append)
    audit.remember(
        _decision(
            decision_point_id="decision-1",
            boundary="event_choice",
            room_context={"room_id": 99},
        )
    )
    result = _decision(
        decision_point_id="decision-2",
        boundary="run_terminal",
        room_context={"room_id": 99},
        run_outcome=False,
    )

    audit.record_action(
        _commit_request(),
        source_branch_id="root",
        result=result,
    )

    assert events[0]["run_result"] is False


def test_speculative_branch_result_does_not_create_root_room_result() -> None:
    events: list[dict] = []
    audit = SelectionAudit(events.append)
    audit.remember(
        _decision(
            decision_point_id="decision-1",
            boundary="combat",
            room_context={"room_id": 7},
        )
    )
    request = {
        **_commit_request(),
        "operation": "emulate_action",
        "branch_id": "branch-1",
        "parent_branch_id": "root",
        "rng_id": 1,
    }
    result = {
        **_decision(
            decision_point_id="branch-decision-1",
            boundary="terminal",
            room_context={"room_id": 7, "outcome": "victory"},
        ),
        "branch_id": "branch-1",
        "parent_branch_id": "root",
        "rng_id": 1,
    }

    audit.record_action(request, source_branch_id="root", result=result)

    assert len(events) == 1
    assert "room_result" not in events[0]
    assert "run_result" not in events[0]
