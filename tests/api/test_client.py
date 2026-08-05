import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts2_training.api.client import ApiProtocolError, TrainingApiClient
from sts2_training.api.transport import FakeTransport


def id_factory(*values: str):
    iterator: Iterator[str] = iter(values)
    return lambda: next(iterator)


def completed_response(operation: str, request_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": "0.5",
        "request_id": request_id,
        "operation": operation,
        "status": "completed",
        **extra,
    }


def decision_fields(branch_id: str = "root") -> dict[str, Any]:
    return {
        "instance_id": "inst-001",
        "branch_id": branch_id,
        "decision_point_id": "decision-002",
        "masked_emulator_dto": {
            "dto_version": "emulator-fca2f06",
            "mask_version": "1.0",
            "legal_actions": [{"action_id": "a-0000"}],
        },
    }


def make_client(response: dict[str, Any], request_id: str) -> tuple[TrainingApiClient, FakeTransport]:
    transport = FakeTransport([response])
    client = TrainingApiClient(transport, request_id_factory=lambda: request_id)
    return client, transport


def test_start_instance_sends_expected_request_and_returns_instance_id() -> None:
    response = completed_response(
        "start_instance",
        "req-001",
        instance_id="inst-001",
    )
    client, transport = make_client(response, "req-001")

    actual = client.start_instance(
        {"instance_type": "combat"},
        timeout_s=1.0,
    )

    assert actual == "inst-001"
    assert transport.requests == [
        {
            "schema_version": "0.5",
            "request_id": "req-001",
            "operation": "start_instance",
            "instance_config": {"instance_type": "combat"},
        }
    ]


def test_start_instance_rejects_mismatched_request_id() -> None:
    response = completed_response(
        "start_instance",
        "req-WRONG",
        instance_id="inst-001",
    )
    client, _ = make_client(response, "req-001")

    with pytest.raises(ApiProtocolError, match="request_id"):
        client.start_instance({"instance_type": "combat"}, timeout_s=1.0)


def test_start_instance_rejects_missing_instance_id() -> None:
    response = completed_response("start_instance", "req-001")
    client, _ = make_client(response, "req-001")

    with pytest.raises(ApiProtocolError, match="instance_id"):
        client.start_instance({"instance_type": "combat"}, timeout_s=1.0)


def test_get_decision_sends_expected_request_and_returns_response() -> None:
    response = completed_response("get_decision", "req-002", **decision_fields())
    client, transport = make_client(response, "req-002")

    actual = client.get_decision("inst-001", "root", timeout_s=1.0)

    assert actual == response
    assert transport.requests == [
        {
            "schema_version": "0.5",
            "request_id": "req-002",
            "operation": "get_decision",
            "instance_id": "inst-001",
            "branch_id": "root",
        }
    ]


def test_get_decision_rejects_mismatched_branch_id() -> None:
    response = completed_response(
        "get_decision",
        "req-002",
        **decision_fields(branch_id="branch-WRONG"),
    )
    client, _ = make_client(response, "req-002")

    with pytest.raises(ApiProtocolError, match="branch_id"):
        client.get_decision("inst-001", "root", timeout_s=1.0)


def test_get_decision_rejected_response_is_returned_for_application_handling() -> None:
    response = {
        "schema_version": "0.5",
        "request_id": "req-002",
        "operation": "get_decision",
        "status": "rejected",
        "instance_id": "inst-001",
        "error": "unknown branch_id",
    }
    client, _ = make_client(response, "req-002")

    assert client.get_decision("inst-001", "root", timeout_s=1.0) == response


def test_commit_action_builds_root_request() -> None:
    response = completed_response("commit_action", "req-003", **decision_fields())
    client, transport = make_client(response, "req-003")

    actual = client.commit_action(
        "inst-001",
        "decision-001",
        "a-0000",
        timeout_s=1.0,
    )

    assert actual == response
    assert transport.requests == [
        {
            "schema_version": "0.5",
            "request_id": "req-003",
            "operation": "commit_action",
            "instance_id": "inst-001",
            "branch_id": "root",
            "rng_id": 0,
            "decision_point_id": "decision-001",
            "action_id": "a-0000",
        }
    ]


def test_emulate_action_builds_request_with_options() -> None:
    response = completed_response(
        "emulate_action",
        "req-004",
        parent_branch_id="root",
        rng_id=1,
        **decision_fields(branch_id="branch-001"),
    )
    client, transport = make_client(response, "req-004")

    actual = client.emulate_action(
        "inst-001",
        "root",
        "branch-001",
        1,
        "decision-001",
        "a-0000",
        timeout_s=1.0,
        simulation_options={
            "stop_condition": "next_decision",
            "max_depth": 1,
            "max_steps": 100,
            "max_time_ms": 5000,
            "max_hypotheses": 1,
        },
    )

    assert actual == response
    assert transport.requests[-1] == {
        "schema_version": "0.5",
        "request_id": "req-004",
        "operation": "emulate_action",
        "instance_id": "inst-001",
        "parent_branch_id": "root",
        "branch_id": "branch-001",
        "rng_id": 1,
        "decision_point_id": "decision-001",
        "action_id": "a-0000",
        "simulation_options": {
            "stop_condition": "next_decision",
            "max_depth": 1,
            "max_steps": 100,
            "max_time_ms": 5000,
            "max_hypotheses": 1,
        },
    }


def test_emulate_action_accepts_running_without_decision_payload() -> None:
    response = {
        "schema_version": "0.5",
        "request_id": "req-004",
        "operation": "emulate_action",
        "status": "running",
        "instance_id": "inst-001",
        "parent_branch_id": "root",
        "branch_id": "branch-001",
        "rng_id": 1,
    }
    client, _ = make_client(response, "req-004")

    assert client.emulate_action(
        "inst-001",
        "root",
        "branch-001",
        1,
        "decision-001",
        "a-0000",
        timeout_s=1.0,
    ) == response


def test_emulate_action_rejects_root_as_new_branch() -> None:
    response = completed_response("emulate_action", "req-004")
    client, _ = make_client(response, "req-004")

    with pytest.raises(ValueError, match="root"):
        client.emulate_action(
            "inst-001",
            "root",
            "root",
            1,
            "decision-001",
            "a-0000",
            timeout_s=1.0,
        )


def test_emulate_action_rejects_non_positive_rng_id() -> None:
    response = completed_response("emulate_action", "req-004")
    client, _ = make_client(response, "req-004")

    with pytest.raises(ValueError, match="positive integer"):
        client.emulate_action(
            "inst-001",
            "root",
            "branch-001",
            0,
            "decision-001",
            "a-0000",
            timeout_s=1.0,
        )


@pytest.mark.parametrize(
    "method_name,operation",
    [
        ("cancel_branches", "cancel_branches"),
        ("release_branches", "release_branches"),
        ("get_branch_status", "get_branch_status"),
    ],
)
def test_branch_batch_operations_build_expected_request(
    method_name: str,
    operation: str,
) -> None:
    response = completed_response(
        operation,
        "req-005",
        instance_id="inst-001",
    )
    client, transport = make_client(response, "req-005")

    method = getattr(client, method_name)
    assert method(
        "inst-001",
        ["branch-001", "branch-002"],
        timeout_s=1.0,
    ) == response

    assert transport.requests == [
        {
            "schema_version": "0.5",
            "request_id": "req-005",
            "operation": operation,
            "instance_id": "inst-001",
            "branch_ids": ["branch-001", "branch-002"],
        }
    ]


def test_branch_batch_operation_rejects_duplicate_ids() -> None:
    response = completed_response("release_branches", "req-005")
    client, _ = make_client(response, "req-005")

    with pytest.raises(ValueError, match="duplicates"):
        client.release_branches(
            "inst-001",
            ["branch-001", "branch-001"],
            timeout_s=1.0,
        )


def test_close_instance_builds_expected_request() -> None:
    response = completed_response(
        "close_instance",
        "req-006",
        instance_id="inst-001",
    )
    client, transport = make_client(response, "req-006")

    assert client.close_instance("inst-001", timeout_s=1.0) == response
    assert transport.requests == [
        {
            "schema_version": "0.5",
            "request_id": "req-006",
            "operation": "close_instance",
            "instance_id": "inst-001",
        }
    ]


def test_client_close_closes_transport() -> None:
    response = completed_response("close_instance", "req-006")
    client, transport = make_client(response, "req-006")

    client.close()

    assert not transport.is_alive()


def test_faulted_response_requires_error_and_fault_kind() -> None:
    response = {
        "schema_version": "0.5",
        "request_id": "req-002",
        "operation": "get_decision",
        "status": "faulted",
        "instance_id": "inst-001",
        "error": "worker failed",
        # fault_kind intentionally missing
    }
    client, _ = make_client(response, "req-002")

    with pytest.raises(ApiProtocolError, match="fault_kind"):
        client.get_decision("inst-001", "root", timeout_s=1.0)


def test_sequential_calls_use_one_new_request_id_each() -> None:
    start_response = completed_response(
        "start_instance",
        "req-001",
        instance_id="inst-001",
    )
    get_response = completed_response("get_decision", "req-002", **decision_fields())
    transport = FakeTransport([start_response, get_response])
    client = TrainingApiClient(
        transport,
        request_id_factory=id_factory("req-001", "req-002"),
    )

    instance_id = client.start_instance(
        {"instance_type": "combat"},
        timeout_s=1.0,
    )
    client.get_decision(instance_id, "root", timeout_s=1.0)

    assert [request["request_id"] for request in transport.requests] == [
        "req-001",
        "req-002",
    ]
