"""`RlServerPool` starts real subprocesses, so these tests do too.

The stub checkout below has the same shape the pool requires (``API/tcp_server.py``
runnable as ``python -m API.tcp_server --host H --port P``), which keeps the spawn, the
readiness wait, and the shutdown on the real code path without needing an Emulator.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from sts2_training.runner.rl_server_pool import (
    RlServerPool,
    free_ports,
    resolve_rl_root,
)

_LISTENING_STUB = """\
import argparse, socket, sys, time
parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--max-message-bytes", type=int, default=None)
args = parser.parse_args()
print("stub starting", args.port, flush=True)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((args.host, args.port))
sock.listen(8)
print("stub listening", args.port, flush=True)
while True:
    conn, _ = sock.accept()
    conn.close()
"""

_DYING_STUB = """\
import sys
sys.stderr.write("stub refused to start\\n")
raise SystemExit(3)
"""

_SLOW_STUB = """\
import time
print("stub is slow", flush=True)
time.sleep(60)
"""


def _make_checkout(directory: Path, body: str) -> Path:
    api = directory / "API"
    api.mkdir(parents=True, exist_ok=True)
    (api / "__init__.py").write_text("", encoding="utf-8")
    (api / "tcp_server.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return directory


class ResolveRootTest(unittest.TestCase):
    def test_missing_root_is_reported(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "no STS2_RL checkout given"):
                resolve_rl_root(None)

    def test_env_var_is_used_when_no_explicit_root_is_given(self) -> None:
        with TemporaryDirectory() as tmp:
            root = _make_checkout(Path(tmp), _LISTENING_STUB)
            with mock.patch.dict(os.environ, {"STS2_RL_ROOT": str(root)}, clear=True):
                self.assertEqual(resolve_rl_root(None), root.resolve())

    def test_directory_without_tcp_server_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "not a usable STS2_RL checkout"):
                resolve_rl_root(tmp)

    def test_valid_checkout_resolves(self) -> None:
        with TemporaryDirectory() as tmp:
            root = _make_checkout(Path(tmp), _LISTENING_STUB)
            self.assertEqual(resolve_rl_root(root), root.resolve())


class FreePortsTest(unittest.TestCase):
    def test_ports_are_distinct(self) -> None:
        ports = free_ports(5)

        self.assertEqual(len(ports), 5)
        self.assertEqual(len(set(ports)), 5)

    def test_non_positive_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "count must be a positive integer"):
            free_ports(0)


class PoolLifecycleTest(unittest.TestCase):
    def _pool(self, body: str, *, ports, log_dir: Path, **kwargs) -> RlServerPool:
        root = _make_checkout(log_dir / "rl", body)
        return RlServerPool(ports=ports, root=root, log_dir=log_dir, **kwargs)

    def test_servers_start_listening_and_stop_afterwards(self) -> None:
        with TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            ports = free_ports(2)
            pool = self._pool(_LISTENING_STUB, ports=ports, log_dir=log_dir)

            with pool as started:
                self.assertEqual(started, ports)
                for port in ports:
                    with socket.create_connection(("127.0.0.1", port), timeout=2):
                        pass
                pids = [server.pid for server in pool.servers]
                for server in pool.servers:
                    self.assertTrue(server.log_path.is_file())

            for pid in pids:
                self.assertFalse(_process_alive(pid), f"pid {pid} survived shutdown")
            for port in ports:
                self.assertFalse(_port_open(port), "port still accepting after shutdown")

    def test_a_server_that_dies_during_startup_reports_its_log(self) -> None:
        with TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            pool = self._pool(_DYING_STUB, ports=free_ports(1), log_dir=log_dir)

            with self.assertRaises(RuntimeError) as caught:
                with pool:
                    pass

            message = str(caught.exception)
            self.assertIn("exited during startup", message)
            self.assertIn("stub refused to start", message)

    def test_a_server_that_never_listens_times_out_with_its_log(self) -> None:
        with TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            pool = self._pool(
                _SLOW_STUB, ports=free_ports(1), log_dir=log_dir, startup_timeout_s=1.0
            )

            with self.assertRaises(TimeoutError) as caught:
                with pool:
                    pass

            self.assertIn("did not start within", str(caught.exception))
            self.assertIn("stub is slow", str(caught.exception))

    def test_servers_are_stopped_when_the_body_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            pool = self._pool(_LISTENING_STUB, ports=free_ports(1), log_dir=log_dir)

            with self.assertRaises(RuntimeError):
                with pool:
                    pids = [server.pid for server in pool.servers]
                    raise RuntimeError("evaluation blew up")

            for pid in pids:
                self.assertFalse(_process_alive(pid), "an exception must not leak servers")

    def test_duplicate_ports_are_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "ports must be unique"):
                self._pool(_LISTENING_STUB, ports=[8765, 8765], log_dir=Path(tmp))

    def test_empty_ports_are_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "ports must not be empty"):
                self._pool(_LISTENING_STUB, ports=[], log_dir=Path(tmp))


def _port_open(port: int) -> bool:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def _process_alive(pid: int) -> bool:
    if sys.platform == "win32":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


if __name__ == "__main__":
    unittest.main()


class EvaluateWholeRunWiringTest(unittest.TestCase):
    """`--start-rl-servers` must hand the started ports to the evaluation, and only then."""

    def _tool(self):
        import importlib.util

        path = Path(__file__).resolve().parents[2] / "tools" / "evaluate_whole_run.py"
        spec = importlib.util.spec_from_file_location("_ewr_under_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_pool_is_not_used_without_the_flag(self) -> None:
        module = self._tool()
        args = module._parse_args(["--character-id", "IRONCLAD"])  # noqa: SLF001

        with mock.patch.object(module, "RlServerPool") as pool:
            with mock.patch.object(module, "_evaluate", return_value={"ok": True}):
                self.assertEqual(module._run_report(args), {"ok": True})  # noqa: SLF001

        pool.assert_not_called()

    def test_started_ports_reach_the_evaluation(self) -> None:
        module = self._tool()
        args = module._parse_args(  # noqa: SLF001
            ["--character-id", "IRONCLAD", "--start-rl-servers", "3"]
        )
        seen: dict = {}

        async def _capture(namespace):
            seen["ports"] = namespace.ports
            return {"ok": True}

        pool = mock.MagicMock()
        pool.return_value.__enter__.return_value = [1111, 2222, 3333]
        with mock.patch.object(module, "RlServerPool", pool):
            with mock.patch.object(module, "free_ports", return_value=[1111, 2222, 3333]):
                with mock.patch.object(module, "_evaluate", _capture):
                    module._run_report(args)  # noqa: SLF001

        self.assertEqual(seen["ports"], [1111, 2222, 3333])
        self.assertEqual(pool.call_args.kwargs["ports"], [1111, 2222, 3333])
        pool.return_value.__exit__.assert_called_once()

    def test_explicit_ports_are_used_for_the_started_servers(self) -> None:
        module = self._tool()
        args = module._parse_args(  # noqa: SLF001
            [
                "--character-id", "IRONCLAD",
                "--start-rl-servers", "2",
                "--ports", "8801,8802",
            ]
        )
        pool = mock.MagicMock()
        pool.return_value.__enter__.return_value = [8801, 8802]

        with mock.patch.object(module, "RlServerPool", pool):
            with mock.patch.object(module, "free_ports") as chooser:
                with mock.patch.object(module, "_evaluate", return_value={}):
                    module._run_report(args)  # noqa: SLF001

        chooser.assert_not_called()
        self.assertEqual(pool.call_args.kwargs["ports"], [8801, 8802])

    def test_server_count_must_match_explicit_ports(self) -> None:
        module = self._tool()
        args = module._parse_args(  # noqa: SLF001
            [
                "--character-id", "IRONCLAD",
                "--start-rl-servers", "4",
                "--ports", "8801,8802",
            ]
        )

        with mock.patch.object(module, "RlServerPool") as pool:
            with self.assertRaises(SystemExit):
                module._run_report(args)  # noqa: SLF001

        pool.assert_not_called()
