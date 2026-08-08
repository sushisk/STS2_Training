"""Regression coverage for bounded SelectionAudit Decision caching."""

from __future__ import annotations

from sts2_training.api.contract import ApiContract
from sts2_training.selection_log import SelectionAudit


def _decision(instance_id: str, branch_id: str) -> dict:
    return {
        "instance_id": instance_id,
        "branch_id": branch_id,
        "decision_point_id": f"decision-{branch_id}",
        "masked_emulator_dto": {"legal_actions": []},
    }


def test_selection_audit_forget_drops_released_branch_decisions() -> None:
    audit = SelectionAudit(lambda event: None)
    branch_ids = [f"branch-{index}" for index in range(5000)]

    for branch_id in branch_ids:
        audit.remember(_decision("instance", branch_id))

    assert len(audit._decisions) == len(branch_ids)  # noqa: SLF001
    audit.forget("instance", branch_ids)
    assert audit._decisions == {}  # noqa: SLF001


def test_successful_cancel_and_release_evict_cached_decisions() -> None:
    for operation, terminal_status in (
        ("cancel_branches", "cancelled"),
        ("release_branches", "released"),
    ):
        contract = ApiContract(
            client_session_id=f"session-{operation}",
            selection_logger=lambda event: None,
        )
        contract._instance_id = "instance"  # noqa: SLF001
        contract._audit.remember(_decision("instance", "branch-1"))  # noqa: SLF001
        assert ("instance", "branch-1") in contract._audit._decisions  # noqa: SLF001

        request = contract._build_branch_batch_operation(  # noqa: SLF001
            1,
            operation,
            "instance",
            ["branch-1"],
        )
        response = {
            **request,
            "server_epoch": "epoch",
            "status": "completed",
            "branch_statuses": {"branch-1": terminal_status},
        }

        contract._validate_api_response(request, response)  # noqa: SLF001

        assert ("instance", "branch-1") not in contract._audit._decisions  # noqa: SLF001
