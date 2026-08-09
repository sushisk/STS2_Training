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


def _sparse_actions(first: str, second: str) -> list[dict]:
    return [
        {"action_id": first, "action_type": "choice_reward_card", "is_available": True},
        {"action_id": second, "action_type": "choice_reward_skip", "is_available": True},
    ]


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


def test_whole_run_sparse_commit_records_public_action_id_and_preserves_wire_token() -> None:
    events: list[dict] = []
    audit = SelectionAudit(events.append)
    start = _decision(
        decision_point_id="decision-1",
        boundary="reward",
        room_context={"room_id": 7},
        legal_actions=_sparse_actions("1", "3"),
    )
    start["operation"] = "start_instance"
    # Whole Run intentionally omits max_emulate_actions_items in DTO v0.7.
    audit.remember(start)

    # The selected public action is "3", but the deployed Whole Run server consumes
    # ordinal "1" at the commit_action wire boundary. The direct public ID "1" also
    # exists, making this a deliberate collision test rather than a trivial remapping.
    result = _decision(
        decision_point_id="decision-2",
        boundary="reward",
        room_context={"room_id": 7},
        legal_actions=_sparse_actions("5", "8"),
    )
    audit.record_action(
        {**_commit_request(), "action_id": "1"},
        source_branch_id="root",
        result=result,
    )

    assert events[0]["request"]["action_id"] == "1"
    assert events[0]["selected_action_id"] == "3"

    # A successful root commit clears stale Decisions but does not close the instance.
    # The Whole Run ordinal compatibility mode must therefore survive into the next
    # decision; ordinal "0" below labels public action "5".
    audit.record_action(
        {
            **_commit_request(request_id="session-a:3"),
            "request_seq": 3,
            "decision_point_id": "decision-2",
            "action_id": "0",
        },
        source_branch_id="root",
        result=_decision(
            decision_point_id="decision-3",
            boundary="map_select",
            room_context={"room_id": 7},
        ),
    )

    assert events[1]["request"]["action_id"] == "0"
    assert events[1]["selected_action_id"] == "5"


def test_combat_numeric_public_action_id_is_not_treated_as_an_ordinal() -> None:
    events: list[dict] = []
    audit = SelectionAudit(events.append)
    start = _decision(
        decision_point_id="decision-1",
        boundary="combat",
        room_context={"room_id": 7},
        legal_actions=_sparse_actions("1", "3"),
    )
    start.update(
        {
            "operation": "start_instance",
            "max_emulate_actions_items": 64,
        }
    )
    audit.remember(start)

    audit.record_action(
        {**_commit_request(), "action_id": "1"},
        source_branch_id="root",
        result=_decision(
            decision_point_id="decision-2",
            boundary="combat",
            room_context={"room_id": 7},
        ),
    )

    assert events[0]["request"]["action_id"] == "1"
    assert events[0]["selected_action_id"] == "1"
