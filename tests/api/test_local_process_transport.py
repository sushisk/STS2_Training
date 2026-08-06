from __future__ import annotations

from pathlib import Path

import pytest

from sts2_training.api.local_process_transport import LocalProcessTransport
from sts2_training.api.transport import (
    RuntimeExitedError,
    TransportClosedError,
    TransportError,
)


class FakeServerProcess:
    instances: list["FakeServerProcess"] = []

    def __init__(self, *, repo_root: Path, request_timeout_s: float) -> None:
        self.repo_root = repo_root
        self.request_timeout_s = request_timeout_s
        self.pid = 4321
        self.alive = True
        self.close_calls = 0
        self.calls: list[tuple[dict, float]] = []
        self.response: object = {"status": "completed"}
        self.error: BaseException | None = None
        type(self).instances.append(self)

    def is_alive(self) -> bool:
        return self.alive

    def call(self, payload: dict):
        self.calls.append((dict(payload), self.request_timeout_s))
        if self.error is not None:
            raise self.error
        return self.response

    def close(self) -> None:
        self.close_calls += 1
        self.alive = False


@pytest.fixture
def fake_rl_root(tmp_path: Path) -> Path:
    api_dir = tmp_path / "API"
    api_dir.mkdir()
    (api_dir / "api_runtime.py").write_text(
        "# fake runtime used by transport unit tests\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture(autouse=True)
def fake_server_process(monkeypatch: pytest.MonkeyPatch):
    FakeServerProcess.instances.clear()
    monkeypatch.setattr(
        LocalProcessTransport,
        "_load_server_process_class",
        classmethod(lambda cls, repo_root: FakeServerProcess),
    )


def test_transport_starts_runtime_and_temporarily_applies_call_timeout(
    fake_rl_root: Path,
) -> None:
    transport = LocalProcessTransport(
        repo_root=fake_rl_root,
        default_timeout_s=10.0,
    )
    server = FakeServerProcess.instances[-1]

    response = transport.call({"request_id": "req-001"}, timeout_s=2.5)

    assert response == {"status": "completed"}
    assert transport.repo_root == fake_rl_root.resolve()
    assert transport.pid == 4321
    assert transport.is_alive()
    assert server.calls == [({"request_id": "req-001"}, 2.5)]
    assert server.request_timeout_s == 10.0


def test_transport_uses_sts2_rl_root_environment_variable(
    fake_rl_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STS2_RL_ROOT", str(fake_rl_root))

    transport = LocalProcessTransport(default_timeout_s=1.0)

    assert transport.repo_root == fake_rl_root.resolve()
    transport.close()


def test_transport_requires_valid_rl_runtime_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STS2_RL_ROOT", raising=False)
    with pytest.raises(ValueError, match="STS2_RL_ROOT"):
        LocalProcessTransport(default_timeout_s=1.0)

    with pytest.raises(FileNotFoundError, match="api_runtime.py"):
        LocalProcessTransport(repo_root=tmp_path, default_timeout_s=1.0)


def test_transport_rejects_non_positive_timeouts(fake_rl_root: Path) -> None:
    with pytest.raises(ValueError, match="default_timeout_s"):
        LocalProcessTransport(repo_root=fake_rl_root, default_timeout_s=0)

    transport = LocalProcessTransport(repo_root=fake_rl_root, default_timeout_s=1.0)
    with pytest.raises(ValueError, match="timeout_s"):
        transport.call({}, timeout_s=0)
    transport.close()


def test_transport_reports_runtime_exit_before_request(fake_rl_root: Path) -> None:
    transport = LocalProcessTransport(repo_root=fake_rl_root, default_timeout_s=1.0)
    server = FakeServerProcess.instances[-1]
    server.alive = False

    with pytest.raises(RuntimeExitedError, match="not alive"):
        transport.call({}, timeout_s=1.0)


@pytest.mark.parametrize(
    "error, expected_exception, message",
    [
        (OSError("pipe failed"), TransportError, "communication failed"),
        (RuntimeError("out of order"), TransportError, "out of order"),
    ],
)
def test_transport_translates_runtime_call_failures(
    fake_rl_root: Path,
    error: BaseException,
    expected_exception: type[BaseException],
    message: str,
) -> None:
    transport = LocalProcessTransport(repo_root=fake_rl_root, default_timeout_s=1.0)
    server = FakeServerProcess.instances[-1]
    server.error = error

    with pytest.raises(expected_exception, match=message):
        transport.call({}, timeout_s=1.0)


def test_transport_rejects_non_dict_runtime_response(fake_rl_root: Path) -> None:
    transport = LocalProcessTransport(repo_root=fake_rl_root, default_timeout_s=1.0)
    server = FakeServerProcess.instances[-1]
    server.response = ["not", "a", "dict"]

    with pytest.raises(TransportError, match="non-dict"):
        transport.call({}, timeout_s=1.0)


def test_transport_close_is_idempotent(fake_rl_root: Path) -> None:
    transport = LocalProcessTransport(repo_root=fake_rl_root, default_timeout_s=1.0)
    server = FakeServerProcess.instances[-1]

    transport.close()
    transport.close()

    assert server.close_calls == 1
    assert not transport.is_alive()
    with pytest.raises(TransportClosedError, match="closed"):
        transport.call({}, timeout_s=1.0)
