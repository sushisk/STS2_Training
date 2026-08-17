"""Shared subprocess management for integration tests paired against a real
STS2_RL checkout (`STS2_RL_ROOT`) - see `_paired_rl_server_v07.py` for what the
spawned server actually is (the real `RLApiServer`, not a stub).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HOST = "127.0.0.1"

_SERVER_HELPER = Path(__file__).with_name("_paired_rl_server_v07.py")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def start_paired_rl(skip):
    """`skip` is the caller's `pytest.skip` (kept as a parameter rather than an
    import here so this module has no hard pytest dependency beyond what callers
    already bring).

    The server's stdout/stderr are redirected to a temp file, not `PIPE`: a real
    Combat episode that actually concludes can log a lot (engine warnings/stack
    traces included) on a call the test never reads from mid-run, and an unread
    `PIPE` has a bounded OS buffer - once full, the child blocks on its next write
    and every in-flight request silently hangs until the test's own timeout, which
    looks exactly like a server-side deadlock from the client side.
    """
    root_value = os.environ.get("STS2_RL_ROOT")
    if not root_value:
        skip("set STS2_RL_ROOT to the paired STS2_RL checkout")
    root = Path(root_value).resolve()
    if not (root / "API" / "tcp_server.py").is_file():
        skip(f"STS2_RL_ROOT is not a usable RL checkout: {root}")

    port = free_port()
    log_fd, log_path = tempfile.mkstemp(prefix="paired_rl_", suffix=".log")
    log_file = os.fdopen(log_fd, "w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [sys.executable, str(_SERVER_HELPER), str(root), HOST, str(port)],
            cwd=root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except BaseException:
        log_file.close()
        try:
            os.remove(log_path)
        except OSError:
            pass
        raise
    log_file.close()  # the child holds its own duplicated handle from here
    process.paired_rl_log_path = log_path  # type: ignore[attr-defined]

    def _read_log() -> str:
        try:
            return Path(log_path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"<could not read {log_path}: {exc}>"

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = _read_log()
            stop_paired_rl(process)
            raise AssertionError(f"paired RL exited before startup:\n{output}")
        try:
            with socket.create_connection((HOST, port), timeout=0.2):
                return process, port
        except OSError:
            time.sleep(0.1)

    output = _read_log()
    stop_paired_rl(process)
    raise AssertionError(f"paired RL did not start within 30s:\n{output}")


def start_production_rl(skip, *, max_message_bytes: int | None = None):
    """Like `start_paired_rl()`, but launches the REAL `API.tcp_server` CLI directly
    instead of `_paired_rl_server_v07.py`.

    `_paired_rl_server_v07.py` is deliberately test-only: it hardcodes a tiny
    `max_message_bytes=4096` (so `test_paired_rl_v07.py` can trip the boundary on
    purpose) and a small `max_branches=16` cap, and installs a fault-injection hook for
    its own integration test. None of that belongs in a real data-collection run - a
    search wide enough to matter (e.g. Oracle collect's `beam_width=32`,
    `exhaustive_root_actions=True`) routinely creates 200-350+ branches per search, whose
    `cancel_branches` cleanup request alone exceeds 4KB, producing harmless-but-noisy
    `TransportError: 'message_too_large'` failures that were previously deferred as
    non-blocking (only affects post-search cleanup, not collected data) - see
    `Outputs/reports`-equivalent project memory. This helper exists for callers (Oracle
    collect batch scripts, etc.) that actually need production-scale limits.

    `max_message_bytes=None` uses `API.tcp_server`'s own `DEFAULT_MAX_MESSAGE_BYTES`
    (5 MiB as of the TCP max-message-bytes raise) rather than re-hardcoding that number
    here.
    """
    root_value = os.environ.get("STS2_RL_ROOT")
    if not root_value:
        skip("set STS2_RL_ROOT to the paired STS2_RL checkout")
    root = Path(root_value).resolve()
    if not (root / "API" / "tcp_server.py").is_file():
        skip(f"STS2_RL_ROOT is not a usable RL checkout: {root}")

    port = free_port()
    log_fd, log_path = tempfile.mkstemp(prefix="production_rl_", suffix=".log")
    log_file = os.fdopen(log_fd, "w", encoding="utf-8")
    command = [sys.executable, "-m", "API.tcp_server", "--host", HOST, "--port", str(port)]
    if max_message_bytes is not None:
        command += ["--max-message-bytes", str(max_message_bytes)]
    try:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except BaseException:
        log_file.close()
        try:
            os.remove(log_path)
        except OSError:
            pass
        raise
    log_file.close()  # the child holds its own duplicated handle from here
    process.paired_rl_log_path = log_path  # type: ignore[attr-defined]

    def _read_log() -> str:
        try:
            return Path(log_path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"<could not read {log_path}: {exc}>"

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = _read_log()
            stop_paired_rl(process)
            raise AssertionError(f"production RL exited before startup:\n{output}")
        try:
            with socket.create_connection((HOST, port), timeout=0.2):
                return process, port
        except OSError:
            time.sleep(0.1)

    output = _read_log()
    stop_paired_rl(process)
    raise AssertionError(f"production RL did not start within 30s:\n{output}")


def stop_paired_rl(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    log_path = getattr(process, "paired_rl_log_path", None)
    if log_path is not None:
        try:
            os.remove(log_path)
        except OSError:
            pass
