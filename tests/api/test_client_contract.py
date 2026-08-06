from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

import pytest

from sts2_training.api.client import (
    ApiProtocolError,
    RequestFaultedError,
    RequestRejectedError,
    TrainingApiClient,
)
from sts2_training.api.transport import FakeTransport


def ids() -> Any:
    counter = itertools.count(1)
    return lambda: f"req-{next(counter):03d}"


def envelope(
    request_id: str,
    operation: str,
    status: str = "completed",
    **fields: Any,
) -> dict[str, Any]:
    return {
        "schema_version": "0.5",
        "request_id": request_id,
        "operation": operation,
        "status": status,
        **fields,
    }


def start_response(request_id: str = "req-001") -> dict[str, Any]:
    return envelope(
        request_id,
        "start_instance",
        instance_id="inst-001",
    )


def decision_response(
    request_id: str,
    operation: str,
    *,
    branch_id: str = "root",
    status: str = "completed",
    **fields: Any,
) -> dict[str, Any]:
    return envelope(
        request_id,
        operation,
        status,
        instance_id="inst-001",
        branch_id=branch_id,
        decision_point_id=f"decision-{request_id}",
        masked_emulator_dto={"legal_actions": []},
        **fields,
    )


def started_client(*responses: dict[str, Any]) -> tuple[TrainingApiClient, FakeTransport]:
    transport = FakeTransport([start_response(), *responses])
    client = TrainingApiClient(transport, request_id_factory=ids())
    assert client.start_instance({"instance_type": "combat"}, timeout_s=1.0) == "inst-001"
    return client, transport


class StaticTransport:
    def __init__(self, response: object) -> None:
        self.response = response
        self.closed = False

    def call(self, request: Mapping[str, Any], *, timeout_s: float) -> object:
        return self.response

    def is_alive(self) -> bool:
        return not self.closed

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "0.4", "schema_version"),
        ("request_id", "req-wrong", "request_id"),
        ("operation", "get_decision", "operation"),
        ("status", "unknown", "unknown status"),
    ],
)
def test_start_rejects_invalid_response_envelope(
    field: str,
    value: str,
    message: str,
) -> None:
    response = start_response()
    response[field] = value
    client = TrainingApiClient(FakeTransport([response]), request_id_factory=ids())

    with pytest.raises(ApiProtocolError, match=message):
        client.start_instance({"instance_type": "combat"}, timeout_s=1.0)


@pytest.mark.parametrize("status_value", [None, "", 1])
def test_status_must_be_a_non_empty_string(status_value: object) -> None:
    response = start_response()
    response["status"] = status_value
    client = TrainingApiClient(FakeTransport([response]), request_id_factory=ids())

    with pytest.raises(ApiProtocolError, match="status"):
        client.start_instance({"instance_type": "combat"}, timeout_s=1.0)


def test_response_must_be_a_dictionary() -> None:
    client = TrainingApiClient(StaticTransport([]), request_id_factory=ids())  # type: ignore[arg-type]

    with pytest.raises(ApiProtocolError, match="dictionary"):
        client.start_instance({"instance_type": "combat"}, timeout_s=1.0)


@pytest.mark.parametrize("request_id", [None, "", 1])
def test_request_id_factory_must_return_non_empty_string(request_id: object) -> None:
    client = TrainingApiClient(
        FakeTransport([]),
        request_id_factory=lambda: request_id,  # type: ignore[return-value]
    )

    with pytest.raises(ValueError, match="request_id"):
        client.start_instance({"instance_type": "combat"}, timeout_s=1.0)


@pytest.mark.parametrize(
    ("status", "fields", "error_type", "message"),
    [
        ("rejected", {}, ApiProtocolError, "error"),
        ("rejected", {"error": "bad config"}, RequestRejectedError, "bad config"),
        ("faulted", {"error": "worker died"}, ApiProtocolError, "fault_kind"),
        (
            "faulted",
            {"error": "worker died", "fault_kind": "worker_process_crash"},
            RequestFaultedError,
            "worker died",
        ),
    ],
)
def test_failure_responses_require_contract_fields_and_raise_distinct_errors(
    status: str,
    fields: dict[str, Any],
    error_type: type[Exception],
    message: str,
) -> None:
    response = envelope("req-001", "start_instance", status, **fields)
    client = TrainingApiClient(FakeTransport([response]), request_id_factory=ids())

    with pytest.raises(error_type, match=message):
        client.start_instance({"instance_type": "combat"}, timeout_s=1.0)


def test_instance_response_must_echo_instance_id_even_when_rejected() -> None:
    client, _ = started_client(
        envelope(
            "req-002",
            "get_decision",
            "rejected",
            instance_id="inst-other",
            error="stale decision",
        )
    )

    with pytest.raises(ApiProtocolError, match="instance_id"):
        client.get_decision("inst-001", timeout_s=1.0)


@pytest.mark.parametrize(
    ("response_patch", "message"),
    [
        ({"decision_point_id": None}, "decision_point_id"),
        ({"masked_emulator_dto": []}, "masked_emulator_dto"),
        ({"branch_id": "branch-other"}, "branch_id"),
    ],
)
def test_completed_get_decision_validates_correlated_payload(
    response_patch: dict[str, Any],
    message: str,
) -> None:
    response = decision_response("req-002", "get_decision")
    response.update(response_patch)
    client, _ = started_client(response)

    with pytest.raises(ApiProtocolError, match=message):
        client.get_decision("inst-001", timeout_s=1.0)


def test_non_completed_get_decision_does_not_require_decision_payload() -> None:
    response = envelope(
        "req-002",
        "get_decision",
        "running",
        instance_id="inst-001",
    )
    client, _ = started_client(response)

    assert client.get_decision("inst-001", timeout_s=1.0)["status"] == "running"


@pytest.mark.parametrize(
    ("decision_point_id", "action_id", "message"),
    [
        ("", "action-1", "decision_point_id"),
        ("decision-1", "", "action_id"),
    ],
)
def test_commit_action_rejects_empty_identifiers_before_transport(
    decision_point_id: str,
    action_id: str,
    message: str,
) -> None:
    client, transport = started_client()

    with pytest.raises(ValueError, match=message):
        client.commit_action(
            "inst-001",
            decision_point_id,
            action_id,
            timeout_s=1.0,
        )
    assert len(transport.requests) == 1


def test_commit_action_sends_root_branch_and_rng_zero() -> None:
    client, transport = started_client(
        decision_response("req-002", "commit_action")
    )

    client.commit_action(
        "inst-001",
        "decision-1",
        "action-1",
        timeout_s=2.5,
    )

    assert transport.requests[-1] == {
        "schema_version": "0.5",
        "request_id": "req-002",
        "operation": "commit_action",
        "instance_id": "inst-001",
        "branch_id": "root",
        "rng_id": 0,
        "decision_point_id": "decision-1",
        "action_id": "action-1",
    }
    assert transport.timeouts[-1] == 2.5


@pytest.mark.parametrize("rng_id", [0, -1, True, 1.5, "1"])
def test_emulate_action_requires_positive_non_boolean_integer_rng(rng_id: object) -> None:
    client, transport = started_client()

    with pytest.raises(ValueError, match="rng_id"):
        client.emulate_action(
            "inst-001",
            "root",
            "branch-1",
            rng_id,  # type: ignore[arg-type]
            "decision-1",
            "action-1",
            timeout_s=1.0,
        )
    assert len(transport.requests) == 1


def test_emulate_action_rejects_root_as_new_branch() -> None:
    client, transport = started_client()

    with pytest.raises(ValueError, match="must not be 'root'"):
        client.emulate_action(
            "inst-001",
            "root",
            "root",
            1,
            "decision-1",
            "action-1",
            timeout_s=1.0,
        )
    assert len(transport.requests) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_depth", 0),
        ("max_steps", -1),
        ("max_time_ms", True),
        ("max_hypotheses", 1.5),
    ],
)
def test_emulate_action_validates_positive_simulation_limits(
    field: str,
    value: object,
) -> None:
    client, transport = started_client()

    with pytest.raises(ValueError, match=field):
        client.emulate_action(
            "inst-001",
            "root",
            "branch-1",
            1,
            "decision-1",
            "action-1",
            timeout_s=1.0,
            simulation_options={field: value},
        )
    assert len(transport.requests) == 1


def test_emulate_action_rejects_empty_stop_condition() -> None:
    client, _ = started_client()

    with pytest.raises(ValueError, match="stop_condition"):
        client.emulate_action(
            "inst-001",
            "root",
            "branch-1",
            1,
            "decision-1",
            "action-1",
            timeout_s=1.0,
            simulation_options={"stop_condition": ""},
        )


def test_emulate_action_sends_options_and_validates_correlations() -> None:
    response = decision_response(
        "req-002",
        "emulate_action",
        branch_id="branch-1",
        parent_branch_id="root",
        rng_id=3,
    )
    client, transport = started_client(response)
    options = {"stop_condition": "next_decision", "max_steps": 8}

    result = client.emulate_action(
        "inst-001",
        "root",
        "branch-1",
        3,
        "decision-1",
        "action-1",
        timeout_s=1.0,
        simulation_options=options,
    )

    assert result["status"] == "completed"
    assert transport.requests[-1]["simulation_options"] == options
    assert transport.requests[-1]["simulation_options"] is not options


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("branch_id", "branch-other"),
        ("parent_branch_id", "branch-other"),
        ("rng_id", 4),
    ],
)
def test_emulate_action_rejects_mismatched_response_correlations(
    field: str,
    value: object,
) -> None:
    response = decision_response(
        "req-002",
        "emulate_action",
        branch_id="branch-1",
        parent_branch_id="root",
        rng_id=3,
    )
    response[field] = value
    client, _ = started_client(response)

    with pytest.raises(ApiProtocolError, match=field):
        client.emulate_action(
            "inst-001",
            "root",
            "branch-1",
            3,
            "decision-1",
            "action-1",
            timeout_s=1.0,
        )


@pytest.mark.parametrize("status", ["queued", "running"])
def test_queued_or_running_emulation_does_not_require_decision_payload(status: str) -> None:
    response = envelope(
        "req-002",
        "emulate_action",
        status,
        instance_id="inst-001",
        branch_id="branch-1",
    )
    client, _ = started_client(response)

    result = client.emulate_action(
        "inst-001",
        "root",
        "branch-1",
        1,
        "decision-1",
        "action-1",
        timeout_s=1.0,
    )

    assert result["status"] == status


@pytest.mark.parametrize(
    ("method_name", "operation"),
    [
        ("cancel_branches", "cancel_branches"),
        ("release_branches", "release_branches"),
        ("get_branch_status", "get_branch_status"),
    ],
)
def test_branch_batch_operations_send_expected_request(
    method_name: str,
    operation: str,
) -> None:
    response = envelope(
        "req-002",
        operation,
        instance_id="inst-001",
    )
    client, transport = started_client(response)

    result = getattr(client, method_name)(
        "inst-001",
        ["branch-1", "branch-2"],
        timeout_s=1.0,
    )

    assert result["status"] == "completed"
    assert transport.requests[-1]["branch_ids"] == ["branch-1", "branch-2"]


@pytest.mark.parametrize(
    ("branch_ids", "error_type", "message"),
    [
        ("branch-1", TypeError, "sequence"),
        (b"branch-1", TypeError, "sequence"),
        ([], ValueError, "must not be empty"),
        (["root"], ValueError, "root"),
        (["branch-1", "branch-1"], ValueError, "duplicates"),
        ([""], ValueError, "branch_id"),
    ],
)
def test_branch_batch_operations_reject_invalid_branch_ids(
    branch_ids: object,
    error_type: type[Exception],
    message: str,
) -> None:
    client, transport = started_client()

    with pytest.raises(error_type, match=message):
        client.cancel_branches(
            "inst-001",
            branch_ids,  # type: ignore[arg-type]
            timeout_s=1.0,
        )
    assert len(transport.requests) == 1


def test_active_instance_id_rejects_cross_instance_requests() -> None:
    client, transport = started_client()

    with pytest.raises(ValueError, match="active client instance"):
        client.get_decision("inst-other", timeout_s=1.0)
    assert len(transport.requests) == 1


def test_close_instance_does_not_clear_state_when_status_is_not_completed() -> None:
    response = envelope(
        "req-002",
        "close_instance",
        "running",
        instance_id="inst-001",
    )
    client, _ = started_client(response)

    with pytest.raises(ApiProtocolError, match="unexpected status"):
        client.close_instance("inst-001", timeout_s=1.0)
    assert client.instance_id == "inst-001"


def test_close_delegates_to_transport() -> None:
    transport = FakeTransport([])
    client = TrainingApiClient(transport)

    client.close()

    assert not transport.is_alive()
