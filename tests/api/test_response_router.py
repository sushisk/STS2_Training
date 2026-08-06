import pytest

from sts2_training.api.response_router import ResponseRouter


def test_accept_then_pop_returns_response_once() -> None:
    router = ResponseRouter()
    router.accept(1, {"request_id": "req-1"})
    assert router.pop(1) == {"request_id": "req-1"}
    assert router.pop(1) is None


def test_expired_late_response_is_discarded() -> None:
    router = ResponseRouter()
    router.expire(1)
    assert router.pop(1) is None
    router.accept(1, {"request_id": "req-1"})
    assert router.pop(1) is None


def test_expire_removes_already_pending_response() -> None:
    router = ResponseRouter()
    router.accept(1, {"request_id": "req-1"})
    router.expire(1)
    assert router.pop(1) is None


def test_duplicate_response_is_protocol_error() -> None:
    router = ResponseRouter()
    router.accept(1, {"request_id": "req-1"})
    with pytest.raises(RuntimeError, match="duplicate response"):
        router.accept(1, {"request_id": "req-1"})
