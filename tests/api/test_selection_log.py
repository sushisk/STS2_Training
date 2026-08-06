from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from sts2_training.api.client import RequestRejectedError, TrainingApiClient
from sts2_training.api.transport import FakeTransport
from sts2_training.selection_log import JsonlSelectionLogger


def _start_response() -> dict:
    return {
        "schema_version": "0.5",
        "request_id": "req-001",
        "operation": "start_instance",
        "status": "completed",
        "instance_id": "inst-001",
    }


def _decision_response(
    *,
    request_id: str,
    operation: str,
    decision_point_id: str,
    boundary: str,
    room_context: dict,
    legal_actions: list[dict],
    branch_id: str = "root",
    **dto_fields: object,
) -> dict:
    return {
        "schema_version": "0.5",
        "request_id": request_id,
        "operation": operation,
        "status": "completed",
        "instance_id": "inst-001",
        "branch_id": branch_id,
        "decision_point_id": decision_point_id,
        "branch_log": [],
        "masked_emulator_dto": {
            "boundary": boundary,
            "room_context": room_context,
            "legal_actions": legal_actions,
            **dto_fields,
        },
    }


def _ids():
    values = iter(["req-001", "req-002", "req-003", "req-004"])
    return lambda: next(values)


def _client(responses: list[dict], events: list[dict]) -> TrainingApiClient:
    return TrainingApiClient(
        FakeTransport(responses),
        request_id_factory=_ids(),
        selection_logger=lambda event: events.append(dict(event)),
    )


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


def test_root_commit_logs_one_record_with_room_result() -> None:
    before = _decision_response(
        request_id="req-002",
        operation="get_decision",
        decision_point_id="d-1",
        boundary="combat",
        room_context={"room_id": 7},
        legal_actions=[{"action_id": "a-1"}, {"action_id": "a-2"}],
    )
    after = _decision_response(
        request_id="req-003",
        operation="commit_action",
        decision_point_id="d-2",
        boundary="map_select",
        room_context={"completed_room_id": 7, "outcome": "victory"},
        legal_actions=[],
    )
    events: list[dict] = []
    client = _client([_start_response(), before, after], events)
    instance_id = client.start_instance({"instance_type": "whole_run"}, timeout_s=1.0)
    client.get_decision(instance_id, timeout_s=1.0)

    client.commit_action(instance_id, "d-1", "a-2", timeout_s=1.0)

    assert len(events) == 1
    event = events[0]
    assert event["received"] == before
    assert event["request"]["action_id"] == "a-2"
    assert event["result"] == after
    assert event["room_result"]["room_context_after"]["outcome"] == "victory"


def test_run_terminal_is_in_same_record() -> None:
    before = _decision_response(
        request_id="req-002",
        operation="get_decision",
        decision_point_id="d-1",
        boundary="event_choice",
        room_context={"room_id": 99},
        legal_actions=[{"action_id": "win"}],
    )
    after = _decision_response(
        request_id="req-003",
        operation="commit_action",
        decision_point_id="d-2",
        boundary="run_terminal",
        room_context={"room_id": 99, "outcome": "victory"},
        legal_actions=[],
        run_result={"victory": True, "score": 1234},
    )
    events: list[dict] = []
    client = _client([_start_response(), before, after], events)
    instance_id = client.start_instance({"instance_type": "whole_run"}, timeout_s=1.0)
    client.get_decision(instance_id, timeout_s=1.0)

    client.commit_action(instance_id, "d-1", "win", timeout_s=1.0)

    assert events[0]["run_result"] == {"victory": True, "score": 1234}


def test_none_run_result_does_not_mark_nonterminal_run() -> None:
    before = _decision_response(
        request_id="req-002",
        operation="get_decision",
        decision_point_id="d-1",
        boundary="combat",
        room_context={"room_id": 7},
        legal_actions=[{"action_id": "a-1"}],
    )
    after = _decision_response(
        request_id="req-003",
        operation="commit_action",
        decision_point_id="d-2",
        boundary="combat",
        room_context={"room_id": 7},
        legal_actions=[{"action_id": "a-2"}],
        run_result=None,
    )
    events: list[dict] = []
    client = _client([_start_response(), before, after], events)
    instance_id = client.start_instance({"instance_type": "whole_run"}, timeout_s=1.0)
    client.get_decision(instance_id, timeout_s=1.0)

    client.commit_action(instance_id, "d-1", "a-1", timeout_s=1.0)

    assert "run_result" not in events[0]
    assert "room_result" not in events[0]


def test_false_run_outcome_is_preserved() -> None:
    before = _decision_response(
        request_id="req-002",
        operation="get_decision",
        decision_point_id="d-1",
        boundary="event_choice",
        room_context={"room_id": 99},
        legal_actions=[{"action_id": "lose"}],
    )
    after = _decision_response(
        request_id="req-003",
        operation="commit_action",
        decision_point_id="d-2",
        boundary="run_terminal",
        room_context={"room_id": 99},
        legal_actions=[],
        run_outcome=False,
    )
    events: list[dict] = []
    client = _client([_start_response(), before, after], events)
    instance_id = client.start_instance({"instance_type": "whole_run"}, timeout_s=1.0)
    client.get_decision(instance_id, timeout_s=1.0)

    client.commit_action(instance_id, "d-1", "lose", timeout_s=1.0)

    assert events[0]["run_result"] is False


def test_emulate_action_logs_parent_decision_and_branch_result() -> None:
    before = _decision_response(
        request_id="req-002",
        operation="get_decision",
        decision_point_id="d-1",
        boundary="combat",
        room_context={"room_id": 7},
        legal_actions=[{"action_id": "a-1"}],
    )
    after = {
        **_decision_response(
            request_id="req-003",
            operation="emulate_action",
            decision_point_id="d-b1",
            boundary="terminal",
            room_context={"room_id": 7, "outcome": "victory"},
            legal_actions=[],
            branch_id="branch-1",
        ),
        "parent_branch_id": "root",
        "rng_id": 1,
    }
    events: list[dict] = []
    client = _client([_start_response(), before, after], events)
    instance_id = client.start_instance({"instance_type": "combat"}, timeout_s=1.0)
    client.get_decision(instance_id, timeout_s=1.0)

    client.emulate_action(
        instance_id,
        parent_branch_id="root",
        branch_id="branch-1",
        rng_id=1,
        decision_point_id="d-1",
        action_id="a-1",
        timeout_s=1.0,
    )

    assert len(events) == 1
    assert events[0]["received"] == before
    assert events[0]["result"] == after
    assert "room_result" not in events[0]


def test_rejected_commit_is_logged_before_exception() -> None:
    before = _decision_response(
        request_id="req-002",
        operation="get_decision",
        decision_point_id="d-1",
        boundary="combat",
        room_context={"room_id": 7},
        legal_actions=[{"action_id": "a-1"}],
    )
    rejected = {
        "schema_version": "0.5",
        "request_id": "req-003",
        "operation": "commit_action",
        "status": "rejected",
        "instance_id": "inst-001",
        "error": "stale decision",
    }
    events: list[dict] = []
    client = _client([_start_response(), before, rejected], events)
    instance_id = client.start_instance({"instance_type": "combat"}, timeout_s=1.0)
    client.get_decision(instance_id, timeout_s=1.0)

    with pytest.raises(RequestRejectedError):
        client.commit_action(instance_id, "d-1", "a-1", timeout_s=1.0)

    assert len(events) == 1
    assert events[0]["result"] == rejected


def test_missing_cached_decision_does_not_invent_room_result() -> None:
    after = _decision_response(
        request_id="req-002",
        operation="commit_action",
        decision_point_id="d-2",
        boundary="map_select",
        room_context={},
        legal_actions=[],
    )
    events: list[dict] = []
    ids = iter(["req-001", "req-002"])
    client = TrainingApiClient(
        FakeTransport([_start_response(), after]),
        request_id_factory=lambda: next(ids),
        selection_logger=lambda event: events.append(dict(event)),
    )
    instance_id = client.start_instance({"instance_type": "whole_run"}, timeout_s=1.0)

    client.commit_action(instance_id, "unknown", "12", timeout_s=1.0)

    assert events[0]["received"] is None
    assert "room_result" not in events[0]
