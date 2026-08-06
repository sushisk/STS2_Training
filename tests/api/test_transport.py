import pytest

from sts2_training.api.transport import FakeTransport, TransportClosedError


def test_fake_transport_records_request_and_timeout() -> None:
    response = {"status": "completed"}
    transport = FakeTransport([response])

    actual = transport.call({"request_id": "req-1"}, timeout_s=1.5)

    assert actual == response
    assert transport.requests == [{"request_id": "req-1"}]
    assert transport.timeouts == [1.5]


def test_fake_transport_close_is_idempotent() -> None:
    transport = FakeTransport([])
    transport.close()
    transport.close()
    assert not transport.is_alive()


def test_fake_transport_rejects_call_after_close() -> None:
    transport = FakeTransport([{"status": "completed"}])
    transport.close()
    with pytest.raises(TransportClosedError):
        transport.call({}, timeout_s=1.0)
