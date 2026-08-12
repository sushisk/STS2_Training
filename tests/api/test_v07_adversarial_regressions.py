from __future__ import annotations

from sts2_training.api.contract import ApiContract, MASK_VERSION, SCHEMA_VERSION
from sts2_training.selection_log import SelectionAudit


def test_emulate_actions_accepts_terminal_completed_result_with_no_legal_moves() -> None:
    contract = ApiContract(client_session_id="session-a")
    request = {
        "items": [
            {
                "parent_branch_id": "root",
                "branch_id": "b1",
                "rng_id": 1,
                "decision_point_id": "d-root-001",
                "action_id": "a-001",
            }
        ]
    }
    response = {
        "status": "completed",
        "branch_results": {
            "b1": {
                "status": "completed",
                "branch_id": "b1",
                "parent_branch_id": "root",
                "rng_id": 1,
                "decision_point_id": "d-b1-terminal",
                "branch_log": [],
                "masked_emulator_dto": {
                    "mask_version": MASK_VERSION,
                    "terminal": True,
                    "outcome": "victory",
                    "legal_actions": [],
                },
            }
        },
    }

    contract._validate_emulate_actions_response(request, response)  # noqa: SLF001


def test_batch_result_is_remembered_for_next_depth_received_audit() -> None:
    events: list[dict] = []
    audit = SelectionAudit(events.append)
    audit.remember(
        {
            "instance_id": "inst-001",
            "branch_id": "root",
            "decision_point_id": "d-root-001",
            "masked_emulator_dto": {"legal_actions": [{"action_id": "a-001"}]},
        }
    )

    audit.record_action(
        {
            "schema_version": SCHEMA_VERSION,
            "client_session_id": "session-a",
            "request_seq": 2,
            "request_id": "session-a:2",
            "operation": "emulate_actions",
            "instance_id": "inst-001",
            "parent_branch_id": "root",
            "branch_id": "b1",
            "rng_id": 1,
            "decision_point_id": "d-root-001",
            "action_id": "a-001",
        },
        source_branch_id="root",
        result={
            "status": "completed",
            "branch_id": "b1",
            "parent_branch_id": "root",
            "rng_id": 1,
            "decision_point_id": "d-b1-001",
            "branch_log": [],
            "masked_emulator_dto": {"legal_actions": [{"action_id": "a-next"}]},
        },
    )

    audit.record_action(
        {
            "schema_version": SCHEMA_VERSION,
            "client_session_id": "session-a",
            "request_seq": 3,
            "request_id": "session-a:3",
            "operation": "emulate_actions",
            "instance_id": "inst-001",
            "parent_branch_id": "b1",
            "branch_id": "c1",
            "rng_id": 1,
            "decision_point_id": "d-b1-001",
            "action_id": "a-next",
        },
        source_branch_id="b1",
        result=None,
        error=TimeoutError("completion uncertain"),
    )

    received = events[-1]["received"]
    assert received is not None
    assert received["instance_id"] == "inst-001"
    assert received["branch_id"] == "b1"
    assert received["decision_point_id"] == "d-b1-001"
