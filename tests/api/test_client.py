import itertools

import pytest

from sts2_training.api.client import (
    ApiProtocolError,
    RequestFaultedError,
    RequestRejectedError,
    TrainingApiClient,
)
from sts2_training.api.transport import FakeTransport


def ids():
    counter = itertools.count(1)
    return lambda: f"req-{next(counter):03d}"


def completed_start(request_id: str = "req-001") -> dict:
    return {
        "schema_version": "0.5",
        "request_id": request_id,
        "operation": "start_instance",
        "status": "completed",
        "instance_id": "inst-001",
    }


def test_start_instance_sends_expected_request_and_stores_instance() -> None:
    transport = FakeTransport([completed_start()])
    client = TrainingApiClient(transport, request_id_factory=ids())
    config = {"instance_type": "combat"}

    instance_id = client.start_instance(config, timeout_s=1.0)

    assert instance_id == "inst-001"
    assert client.instance_id == "inst-001"
    assert transport.requests == [
        {
            "schema_version": "0.5",
            "request_id": "req-001",
            "operation": "start_instance",
            "instance_config": config,
        }
    ]


def test_start_instance_rejects_mismatched_request_id() -> None:
    transport = FakeTransport([completed_start("req-WRONG")])
    client = TrainingApiClient(transport, request_id_factory=ids())
    with pytest.raises(ApiProtocolError, match="request_id"):
        client.start_instance({"instance_type": "combat"}, timeout_s=1.0)


def test_rejected_and_faulted_are_distinct() -> None:
    rejected = {
        "schema_version": "0.5",
        "request_id": "req-001",
        "operation": "start_instance",
        "status": "rejected",
        "error": "bad config",
    }
    with pytest.raises(RequestRejectedError):
        TrainingApiClient(
            FakeTransport([rejected]), request_id_factory=ids()
        ).start_instance({"instance_type": "combat"}, timeout_s=1.0)

    faulted = {
        "schema_version": "0.5",
        "request_id": "req-001",
        "operation": "start_instance",
        "status": "faulted",
        "error": "runtime timeout",
        "fault_kind": "task_timeout",
    }
    with pytest.raises(RequestFaultedError):
        TrainingApiClient(
            FakeTransport([faulted]), request_id_factory=ids()
        ).start_instance({"instance_type": "combat"}, timeout_s=1.0)


def test_get_decision_builds_correlated_request() -> None:
    responses = [
        completed_start(),
        {
            "schema_version": "0.5",
            "request_id": "req-002",
            "operation": "get_decision",
            "status": "completed",
            "instance_id": "inst-001",
            "branch_id": "root",
            "decision_point_id": "d-root-001",
            "masked_emulator_dto": {"legal_actions": []},
        },
    ]
    transport = FakeTransport(responses)
    client = TrainingApiClient(transport, request_id_factory=ids())
    instance_id = client.start_instance({"instance_type": "combat"}, timeout_s=1.0)

    response = client.get_decision(instance_id, timeout_s=1.0)

    assert response["decision_point_id"] == "d-root-001"
    assert transport.requests[-1]["branch_id"] == "root"


def test_commit_action_uses_root_and_rng_zero() -> None:
    responses = [
        completed_start(),
        {
            "schema_version": "0.5",
            "request_id": "req-002",
            "operation": "commit_action",
            "status": "completed",
            "instance_id": "inst-001",
            "branch_id": "root",
            "decision_point_id": "d-root-002",
            "masked_emulator_dto": {"legal_actions": []},
        },
    ]
    transport = FakeTransport(responses)
    client = TrainingApiClient(transport, request_id_factory=ids())
    instance_id = client.start_instance({"instance_type": "combat"}, timeout_s=1.0)

    client.commit_action(
        instance_id,
        "d-root-001",
        "a-001",
        timeout_s=1.0,
    )

    request = transport.requests[-1]
    assert request["branch_id"] == "root"
    assert request["rng_id"] == 0


def test_emulate_action_builds_correlated_request() -> None:
    response = {
        "schema_version": "0.5",
        "request_id": "req-002",
        "operation": "emulate_action",
        "status": "completed",
        "instance_id": "inst-001",
        "branch_id": "branch-001",
        "parent_branch_id": "root",
        "rng_id": 1,
        "decision_point_id": "d-branch-001",
        "masked_emulator_dto": {"legal_actions": []},
    }
    transport = FakeTransport([completed_start(), response])
    client = TrainingApiClient(transport, request_id_factory=ids())
    instance_id = client.start_instance({"instance_type": "combat"}, timeout_s=1.0)

    assert client.emulate_action(
        instance_id,
        parent_branch_id="root",
        branch_id="branch-001",
        rng_id=1,
        decision_point_id="d-root-001",
        action_id="0",
        simulation_options={"stop_condition": "next_decision"},
        timeout_s=2.0,
    ) == response
    assert transport.requests[-1] == {
        "schema_version": "0.5",
        "request_id": "req-002",
        "operation": "emulate_action",
        "instance_id": "inst-001",
        "parent_branch_id": "root",
        "branch_id": "branch-001",
        "rng_id": 1,
        "decision_point_id": "d-root-001",
        "action_id": "0",
        "simulation_options": {"stop_condition": "next_decision"},
    }


def test_emulate_action_rejects_mismatched_branch_response() -> None:
    response = {
        "schema_version": "0.5",
        "request_id": "req-002",
        "operation": "emulate_action",
        "status": "running",
        "instance_id": "inst-001",
        "branch_id": "branch-wrong",
    }
    transport = FakeTransport([completed_start(), response])
    client = TrainingApiClient(transport, request_id_factory=ids())
    instance_id = client.start_instance({"instance_type": "combat"}, timeout_s=1.0)

    with pytest.raises(ApiProtocolError, match="branch_id"):
        client.emulate_action(
            instance_id,
            parent_branch_id="root",
            branch_id="branch-001",
            rng_id=1,
            decision_point_id="d-root-001",
            action_id="0",
            timeout_s=1.0,
        )


@pytest.mark.parametrize(
    ("branch_id", "rng_id", "options", "message"),
    [
        ("root", 1, None, "branch_id"),
        ("branch-001", 0, None, "rng_id"),
        ("branch-001", 1, {"max_time_ms": 0}, "max_time_ms"),
    ],
)
def test_emulate_action_rejects_invalid_request(
    branch_id: str,
    rng_id: int,
    options: dict | None,
    message: str,
) -> None:
    client = TrainingApiClient(FakeTransport([]), request_id_factory=ids())

    with pytest.raises(ValueError, match=message):
        client.emulate_action(
            "inst-001",
            parent_branch_id="root",
            branch_id=branch_id,
            rng_id=rng_id,
            decision_point_id="d-root-001",
            action_id="0",
            simulation_options=options,
            timeout_s=1.0,
        )


def test_close_instance_clears_client_state() -> None:
    responses = [
        completed_start(),
        {
            "schema_version": "0.5",
            "request_id": "req-002",
            "operation": "close_instance",
            "status": "completed",
            "instance_id": "inst-001",
        },
    ]
    client = TrainingApiClient(FakeTransport(responses), request_id_factory=ids())
    instance_id = client.start_instance({"instance_type": "combat"}, timeout_s=1.0)
    client.close_instance(instance_id, timeout_s=1.0)
    assert client.instance_id is None
