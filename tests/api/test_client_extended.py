import itertools

import pytest

from sts2_training.api.client import ApiProtocolError, TrainingApiClient
from sts2_training.api.transport import FakeTransport


def ids():
    counter = itertools.count(1)
    return lambda: f"req-{next(counter):03d}"


def completed_start() -> dict:
    return {
        "schema_version": "0.5",
        "request_id": "req-001",
        "operation": "start_instance",
        "status": "completed",
        "instance_id": "inst-001",
    }


def started_client(*responses: dict):
    transport = FakeTransport([completed_start(), *responses])
    client = TrainingApiClient(transport, request_id_factory=ids())
    instance_id = client.start_instance(
        {"instance_type": "combat"},
        timeout_s=1.0,
    )
    return client, transport, instance_id


def test_emulate_action_sends_correlated_branch_request() -> None:
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
    client, transport, instance_id = started_client(response)

    actual = client.emulate_action(
        instance_id,
        parent_branch_id="root",
        branch_id="branch-001",
        rng_id=1,
        decision_point_id="d-root-001",
        action_id="0",
        simulation_options={
            "max_time_ms": 5_000,
            "stop_condition": "next_decision",
        },
        timeout_s=6.0,
    )

    assert actual == response
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
        "simulation_options": {
            "max_time_ms": 5_000,
            "stop_condition": "next_decision",
        },
    }
    assert transport.timeouts[-1] == 6.0


@pytest.mark.parametrize("rng_id", [0, -1, True, 1.5, "1"])
def test_emulate_action_rejects_invalid_rng_id(rng_id) -> None:
    client = TrainingApiClient(FakeTransport([]), request_id_factory=ids())

    with pytest.raises(ValueError, match="rng_id"):
        client.emulate_action(
            "inst-001",
            parent_branch_id="root",
            branch_id="branch-001",
            rng_id=rng_id,
            decision_point_id="d-root-001",
            action_id="0",
            timeout_s=1.0,
        )


def test_emulate_action_rejects_root_as_child_branch() -> None:
    client = TrainingApiClient(FakeTransport([]), request_id_factory=ids())

    with pytest.raises(ValueError, match="must not be 'root'"):
        client.emulate_action(
            "inst-001",
            parent_branch_id="root",
            branch_id="root",
            rng_id=1,
            decision_point_id="d-root-001",
            action_id="0",
            timeout_s=1.0,
        )


@pytest.mark.parametrize(
    "options, field_name",
    [
        ({"max_depth": 0}, "max_depth"),
        ({"max_steps": -1}, "max_steps"),
        ({"max_time_ms": True}, "max_time_ms"),
        ({"max_hypotheses": 1.5}, "max_hypotheses"),
        ({"stop_condition": ""}, "stop_condition"),
    ],
)
def test_emulate_action_rejects_invalid_simulation_options(
    options: dict,
    field_name: str,
) -> None:
    client = TrainingApiClient(FakeTransport([]), request_id_factory=ids())

    with pytest.raises(ValueError, match=field_name):
        client.emulate_action(
            "inst-001",
            parent_branch_id="root",
            branch_id="branch-001",
            rng_id=1,
            decision_point_id="d-root-001",
            action_id="0",
            simulation_options=options,
            timeout_s=1.0,
        )


@pytest.mark.parametrize(
    "branch_ids, error_type, message",
    [
        ([], ValueError, "must not be empty"),
        (["root"], ValueError, "root cannot"),
        (["branch-001", "branch-001"], ValueError, "duplicates"),
        ("branch-001", TypeError, "sequence of strings"),
        ([""], ValueError, "non-empty string"),
    ],
)
def test_branch_batch_operations_validate_branch_ids(
    branch_ids,
    error_type,
    message: str,
) -> None:
    client = TrainingApiClient(FakeTransport([]), request_id_factory=ids())

    with pytest.raises(error_type, match=message):
        client.release_branches(
            "inst-001",
            branch_ids,
            timeout_s=1.0,
        )


def test_get_decision_rejects_active_instance_mismatch() -> None:
    client, transport, _ = started_client()

    with pytest.raises(ValueError, match="active client instance"):
        client.get_decision("inst-other", timeout_s=1.0)

    assert len(transport.requests) == 1


@pytest.mark.parametrize(
    "response, message",
    [
        (
            {
                "schema_version": "0.4",
                "request_id": "req-002",
                "operation": "get_decision",
                "status": "completed",
                "instance_id": "inst-001",
                "branch_id": "root",
                "decision_point_id": "d-root-001",
                "masked_emulator_dto": {},
            },
            "schema_version",
        ),
        (
            {
                "schema_version": "0.5",
                "request_id": "req-002",
                "operation": "commit_action",
                "status": "completed",
                "instance_id": "inst-001",
                "branch_id": "root",
                "decision_point_id": "d-root-001",
                "masked_emulator_dto": {},
            },
            "operation",
        ),
        (
            {
                "schema_version": "0.5",
                "request_id": "req-002",
                "operation": "get_decision",
                "status": "completed",
                "instance_id": "inst-other",
                "branch_id": "root",
                "decision_point_id": "d-root-001",
                "masked_emulator_dto": {},
            },
            "instance_id",
        ),
        (
            {
                "schema_version": "0.5",
                "request_id": "req-002",
                "operation": "get_decision",
                "status": "mystery",
                "instance_id": "inst-001",
            },
            "unknown status",
        ),
    ],
)
def test_response_envelope_mismatches_are_protocol_errors(
    response: dict,
    message: str,
) -> None:
    client, _, instance_id = started_client(response)

    with pytest.raises(ApiProtocolError, match=message):
        client.get_decision(instance_id, timeout_s=1.0)


@pytest.mark.parametrize(
    "response, message",
    [
        (
            {
                "schema_version": "0.5",
                "request_id": "req-002",
                "operation": "get_decision",
                "status": "completed",
                "instance_id": "inst-001",
                "branch_id": "root",
                "masked_emulator_dto": {},
            },
            "decision_point_id",
        ),
        (
            {
                "schema_version": "0.5",
                "request_id": "req-002",
                "operation": "get_decision",
                "status": "completed",
                "instance_id": "inst-001",
                "branch_id": "root",
                "decision_point_id": "d-root-001",
                "masked_emulator_dto": [],
            },
            "masked_emulator_dto",
        ),
    ],
)
def test_completed_decision_requires_decision_payload(
    response: dict,
    message: str,
) -> None:
    client, _, instance_id = started_client(response)

    with pytest.raises(ApiProtocolError, match=message):
        client.get_decision(instance_id, timeout_s=1.0)
