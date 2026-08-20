"""Start and stop a pool of paired STS2_RL servers for a sharded evaluation.

One RL server cannot be parallelized from the client side: `API/tcp_server.py` serializes
every request from every connection behind a single `_handler_lock`, and the Emulator
lives in one pythonnet/CLR process behind it. Using more than one core therefore means
running more than one server, which is tedious to do by hand for every evaluation.

Three things here are not obvious and are the reason this is a module rather than a couple
of `Popen` calls:

* **Server output goes to a file, never to `PIPE`.** A real run logs a lot, and nothing in
  an evaluation ever reads the server's stdout. An unread `PIPE` has a bounded OS buffer;
  once it fills, the child blocks on its next write and every in-flight request hangs until
  the client's own timeout, which looks exactly like a server-side deadlock. (The same trap
  is documented in `tests/integration/_paired_rl_helpers.py`.)
* **Shutdown kills the process tree, not the server process.** The Whole Run server runs a
  `multiprocessing` worker pool (`Run/worker_pool.py`) whose children hold the CLR handles.
  Terminating only the parent orphans those workers, and they keep a core busy and their
  memory resident for as long as the machine is up.
* **Startup is awaited per port.** CLR initialization is slow and starting several servers
  at once makes it slower, so "spawned" is not "ready". Each port is polled until it
  accepts a connection, and a server that dies during startup reports its own log instead
  of a bare timeout.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Sequence

__all__ = ["RlServer", "RlServerPool", "free_ports", "resolve_rl_root"]

_LOG = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_STARTUP_TIMEOUT_S = 180.0
_LOG_TAIL_CHARS = 4000


@dataclass(frozen=True)
class RlServer:
    port: int
    pid: int
    log_path: Path


def resolve_rl_root(root: str | os.PathLike[str] | None = None) -> Path:
    """Locate the paired STS2_RL checkout, preferring an explicit path over the env var."""

    raw = root if root is not None else os.environ.get("STS2_RL_ROOT")
    if not raw:
        raise ValueError(
            "no STS2_RL checkout given: pass --rl-root or set STS2_RL_ROOT to the paired "
            "STS2_RL checkout (the directory containing API/tcp_server.py)"
        )
    resolved = Path(raw).expanduser().resolve()
    if not (resolved / "API" / "tcp_server.py").is_file():
        raise ValueError(
            f"{resolved} is not a usable STS2_RL checkout: API/tcp_server.py is missing"
        )
    return resolved


def free_ports(count: int, *, host: str = DEFAULT_HOST) -> list[int]:
    """Reserve ``count`` distinct free TCP ports.

    Every socket is held open until all of them are chosen, so one call cannot hand out
    the same port twice. There is still a window between closing them and the servers
    binding; that race is inherent to asking the OS for a free port.
    """

    if count <= 0:
        raise ValueError("count must be a positive integer")
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((host, 0))
            sockets.append(sock)
        return [int(sock.getsockname()[1]) for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def _is_listening(host: str, port: int, timeout_s: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _log_tail(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"<could not read {path}: {exc}>"
    return text[-_LOG_TAIL_CHARS:] if len(text) > _LOG_TAIL_CHARS else text


def _kill_tree(process: subprocess.Popen) -> None:
    """Terminate a server and every process it spawned.

    The RL server's `multiprocessing` workers are separate processes that outlive a plain
    `terminate()` on the parent; on Windows they are only reachable through the process
    tree, so `taskkill /T` is used there and a process group elsewhere.
    """

    if process.poll() is not None:
        return
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                check=False,
                capture_output=True,
            )
        except OSError:  # taskkill missing; fall through to the plain kill below
            _LOG.warning("taskkill unavailable; RL worker processes may survive")
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _LOG.warning("RL server pid %d did not exit after kill", process.pid)


class RlServerPool:
    """Context manager owning one `API.tcp_server` process per port.

    Entering returns the ports in the order they were requested; leaving shuts every
    server's process tree down, including on an exception or KeyboardInterrupt, so an
    interrupted evaluation does not leave Emulator processes running.
    """

    def __init__(
        self,
        *,
        ports: Sequence[int],
        root: str | os.PathLike[str] | None = None,
        host: str = DEFAULT_HOST,
        log_dir: str | os.PathLike[str] | None = None,
        startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
        max_message_bytes: int | None = None,
    ) -> None:
        self._ports = list(ports)
        if not self._ports:
            raise ValueError("ports must not be empty")
        if len(set(self._ports)) != len(self._ports):
            raise ValueError("ports must be unique; one server cannot serve two workers")
        self._root = resolve_rl_root(root)
        self._host = host
        self._log_dir = Path(log_dir).resolve() if log_dir is not None else Path.cwd()
        self._startup_timeout_s = float(startup_timeout_s)
        self._max_message_bytes = max_message_bytes
        self._processes: list[subprocess.Popen] = []
        self.servers: list[RlServer] = []

    def __enter__(self) -> list[int]:
        try:
            self._start_all()
        except BaseException:
            self.stop()
            raise
        return list(self._ports)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.stop()

    def _command(self, port: int) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "API.tcp_server",
            "--host",
            self._host,
            "--port",
            str(port),
        ]
        if self._max_message_bytes is not None:
            command += ["--max-message-bytes", str(self._max_message_bytes)]
        return command

    def _spawn(self, port: int) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / f"rl-server-{port}.log"
        log_file = log_path.open("w", encoding="utf-8")
        popen_kwargs: dict = {}
        if sys.platform != "win32":
            # Its own process group, so shutdown can signal the workers too.
            popen_kwargs["start_new_session"] = True
        try:
            process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                self._command(port),
                cwd=self._root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                **popen_kwargs,
            )
        finally:
            # The child holds its own duplicated handle from here.
            log_file.close()
        self._processes.append(process)
        self.servers.append(RlServer(port=port, pid=process.pid, log_path=log_path))

    def _start_all(self) -> None:
        _LOG.info(
            "starting %d RL server(s) from %s on ports %s",
            len(self._ports),
            self._root,
            ", ".join(str(port) for port in self._ports),
        )
        for port in self._ports:
            self._spawn(port)

        deadline = time.monotonic() + self._startup_timeout_s
        pending = list(zip(self._processes, self.servers))
        while pending:
            for process, server in list(pending):
                if process.poll() is not None:
                    raise RuntimeError(
                        f"RL server on port {server.port} exited during startup "
                        f"(exit code {process.returncode}). Log tail:\n"
                        f"{_log_tail(server.log_path)}"
                    )
                if _is_listening(self._host, server.port):
                    pending.remove((process, server))
            if not pending:
                break
            if time.monotonic() >= deadline:
                stalled = ", ".join(str(server.port) for _process, server in pending)
                first = pending[0][1]
                raise TimeoutError(
                    f"RL server(s) on port(s) {stalled} did not start within "
                    f"{self._startup_timeout_s:.0f}s. Log tail for port {first.port}:\n"
                    f"{_log_tail(first.log_path)}"
                )
            time.sleep(0.2)
        _LOG.info("all %d RL server(s) ready", len(self._ports))

    def stop(self) -> None:
        for process in self._processes:
            _kill_tree(process)
        self._processes.clear()
