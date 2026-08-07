from __future__ import annotations

import pytest

from sts2_training.api.contract import ApiContract
from sts2_training.selection_log import SelectionAudit


def _contract() -> ApiContract:
    contract = ApiContract(client_session_id="session-a")
    contract._accept_start_instance(  # noqa: SLF001
        {"status": "completed", "instance_id": "inst-001"}
    )
    return contract


def _item(parent_branch_id: str, branch_id: str) -> dict:
    return {
        "parent_branch_id": parent_branch_id,
        "branch_id": branch_id,
        "rng_id": 1,
        "decision_point_id": "d-root-001",
        "action_id": "a-001",
    }


@pytest.mark.parametrize(
    "items",
    [
        [_item("root", "b1"), _item("b1", "b2")],
        [_item("b1", "b2"), _item("root", "b1")],
    ],
)
def test_same_batch_parent_dependency_is_rejected_in_any_order(items: list[dict]) -> None:
    contract = _contract()

    with pytest.raises(ValueError, match="created within the same batch"):
        contract._build_emulate_actions(  # noqa: SLF001
            1, "inst-001", items, simulation_options=None
        )


def test_selection_replay_identity_rolls_over_with_request_id() -> None:
    events: list[dict] = []
    audit = SelectionAudit(events.append)

    def record(request_id: str, branch_id: str) -> None:
        audit.record_action(
            {
                "schema_version": "0.7",
                "client_session_id": "session-a",
                "request_seq": int(request_id.rsplit(":", 1)[1]),
                "request_id": request_id,
                "operation": "emulate_actions",
                "instance_id": "inst-001",
                "parent_branch_id": "root",
                "branch_id": branch_id,
                "rng_id": 1,
                "decision_point_id": "d-root-001",
                "action_id": "a-001",
            },
            source_branch_id="root",
            result=None,
            error=TimeoutError("uncertain"),
        )

    record("session-a:1", "b1")
    record("session-a:1", "b2")
    assert audit._selection_request_id == "session-a:1"  # noqa: SLF001
    assert audit._selection_branch_ids == {"b1", "b2"}  # noqa: SLF001

    record("session-a:2", "c1")
    assert audit._selection_request_id == "session-a:2"  # noqa: SLF001
    assert audit._selection_branch_ids == {"c1"}  # noqa: SLF001

    record("session-a:2", "c1")
    assert [event["event"] for event in events] == [
        "selection",
        "selection",
        "selection",
        "selection_recovery",
    ]
