import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts2_training.api.transport import FakeTransport


REQUEST = {
    "schema_version": "0.5",
    "request_id": "req-001",
    "operation": "start_instance",
    "instance_config": {"instance_type": "combat"},
}
RESPONSE = {
    "schema_version": "0.5",
    "request_id": "req-001",
    "operation": "start_instance",
    "status": "completed",
    "instance_id": "inst-001",
}


def make_transport() -> FakeTransport:
    return FakeTransport([RESPONSE])


def test_call_returns_first_response() -> None:
    transport = make_transport()
    assert transport.call(REQUEST, timeout_s=1.0) == RESPONSE


def test_call_records_request() -> None:
    transport = make_transport()
    transport.call(REQUEST, timeout_s=1.0)
    assert transport.requests == [REQUEST]


def test_close_changes_alive_state() -> None:
    transport = make_transport()
    transport.close()
    assert not transport.is_alive()


def test_call_after_close_raises_runtime_error() -> None:
    transport = make_transport()
    transport.close()
    with pytest.raises(RuntimeError):
        transport.call(REQUEST, timeout_s=1.0)
