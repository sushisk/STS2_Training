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
