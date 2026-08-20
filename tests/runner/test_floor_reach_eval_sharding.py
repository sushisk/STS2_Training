"""Multi-server sharding: one RL server per worker.

One RL server cannot be parallelized from the client side. `API/tcp_server.py` holds a
single `_handler_lock` across every connection, and the Emulator lives in one spawned
pythonnet/CLR process behind it, so extra workers against one port only queue. Using more
cores means starting more servers and pinning one worker to each.
"""

from __future__ import annotations

import asyncio
import unittest
from collections import Counter

from sts2_training.runner import floor_reach_eval as fre


class _RecordingConnection:
    """Stands in for `TcpConnection` so the port each worker dials is observable."""

    def __init__(self, *, host: str, port: int, connect_timeout_s: float) -> None:
        del host, connect_timeout_s
        self.port = port

    async def connect(self) -> None:
        return None


class _StubRun:
    """Patches `_run_one` so sharding can be tested without a server."""

    def __init__(self) -> None:
        self.ports: list[int] = []
        self.concurrent = 0
        self.max_concurrent = 0

    async def __call__(self, run_id, *, connection_factory, seed, **kwargs):
        del kwargs
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        connection = connection_factory()
        self.ports.append(getattr(connection, "port", None))
        await asyncio.sleep(0)
        self.concurrent -= 1
        return fre.FloorReachResult(
            run_id=run_id,
            seed=seed,
            max_total_floor=1,
            act_index_at_max=0,
            decisions_made=1,
            decision_source_counts={},
            outcome="defeat",
            error=None,
            elapsed_s=0.0,
        )


class ShardingTest(unittest.IsolatedAsyncioTestCase):
    async def _run(self, monkey: _StubRun, **kwargs):
        original = fre._run_one  # noqa: SLF001
        original_conn = fre.TcpConnection
        original_preflight = fre._require_listening_ports  # noqa: SLF001

        async def _skip_preflight(*args, **kwargs):
            del args, kwargs

        fre._run_one = monkey  # noqa: SLF001
        fre.TcpConnection = _RecordingConnection
        fre._require_listening_ports = _skip_preflight  # noqa: SLF001
        try:
            return await fre.run_floor_reach_eval(
                character_id="IRONCLAD",
                num_runs=kwargs.pop("num_runs", 6),
                **kwargs,
            )
        finally:
            fre._run_one = original  # noqa: SLF001
            fre.TcpConnection = original_conn
            fre._require_listening_ports = original_preflight  # noqa: SLF001

    async def test_each_worker_is_pinned_to_its_own_port(self) -> None:
        stub = _StubRun()

        await self._run(stub, ports=[8765, 8766, 8767])

        # Every run went to one of the three servers, and all three were used.
        self.assertEqual(set(stub.ports), {8765, 8766, 8767})
        self.assertEqual(stub.max_concurrent, 3)

    async def test_concurrency_defaults_to_the_port_count(self) -> None:
        stub = _StubRun()

        await self._run(stub, ports=[8765, 8766, 8767, 8768])

        self.assertEqual(stub.max_concurrent, 4)

    async def test_single_port_stays_serial(self) -> None:
        stub = _StubRun()

        await self._run(stub, port=8765)

        self.assertEqual(stub.max_concurrent, 1)
        self.assertEqual(set(stub.ports), {8765})

    async def test_workers_beyond_the_port_count_warn_about_sharing_a_server(self) -> None:
        stub = _StubRun()

        with self.assertLogs(fre._LOG, level="WARNING") as logs:  # noqa: SLF001
            await self._run(stub, ports=[8765, 8766], concurrency=4)

        message = "\n".join(logs.output)
        self.assertIn("serialize behind its request lock", message)
        # Sharing is allowed, just reported: the two servers still get even shares.
        self.assertEqual(Counter(stub.ports)[8765], Counter(stub.ports)[8766])

    async def test_duplicate_ports_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ports must be unique"):
            await fre.run_floor_reach_eval(
                character_id="IRONCLAD", num_runs=1, ports=[8765, 8765]
            )

    async def test_invalid_port_is_rejected(self) -> None:
        for bad in ([0], [70000], ["8765"], [True]):
            with self.subTest(ports=bad):
                with self.assertRaisesRegex(ValueError, "ports must contain TCP port"):
                    await fre.run_floor_reach_eval(
                        character_id="IRONCLAD", num_runs=1, ports=bad
                    )


class PortListParsingTest(unittest.TestCase):
    def test_parses_and_rejects(self) -> None:
        import argparse

        from sts2_training.runner._cli import _port_list

        self.assertEqual(_port_list("8765,8766, 8767"), [8765, 8766, 8767])
        with self.assertRaises(argparse.ArgumentTypeError):
            _port_list("8765,8765")
        with self.assertRaises(argparse.ArgumentTypeError):
            _port_list("")


if __name__ == "__main__":
    unittest.main()


class PortPreflightTest(unittest.IsolatedAsyncioTestCase):
    """A missing server must be named up front, not absorbed as per-run errors."""

    async def _listening_port(self) -> tuple[int, asyncio.AbstractServer]:
        def _hang_up(_reader, writer) -> None:
            writer.close()

        server = await asyncio.start_server(_hang_up, "127.0.0.1", 0)
        self.addCleanup(server.close)
        return server.sockets[0].getsockname()[1], server

    async def test_missing_server_is_reported_before_any_run_starts(self) -> None:
        live_port, server = await self._listening_port()
        dead_port, dead_server = await self._listening_port()
        dead_server.close()

        stub = _StubRun()
        try:
            with self.assertRaises(ConnectionError) as caught:
                await fre.run_floor_reach_eval(
                    character_id="IRONCLAD",
                    num_runs=4,
                    ports=[live_port, dead_port],
                    connect_timeout_s=1.0,
                )
        finally:
            server.close()

        message = str(caught.exception)
        self.assertIn(str(dead_port), message)
        # The port that answered is named as such, not listed as missing.
        self.assertIn(f"only {live_port} answered", message)
        # Point at the flag that removes the problem entirely.
        self.assertIn("--start-rl-servers 2", message)
        self.assertEqual(stub.ports, [])

    async def test_all_ports_listening_passes_the_preflight(self) -> None:
        port_a, server_a = await self._listening_port()
        port_b, server_b = await self._listening_port()
        stub = _StubRun()
        original = fre._run_one  # noqa: SLF001
        fre._run_one = stub  # noqa: SLF001
        try:
            results = await fre.run_floor_reach_eval(
                character_id="IRONCLAD",
                num_runs=2,
                ports=[port_a, port_b],
                connect_timeout_s=1.0,
            )
        finally:
            fre._run_one = original  # noqa: SLF001
            for server in (server_a, server_b):
                server.close()

        self.assertEqual(len(results), 2)

    async def test_preflight_is_skipped_for_an_injected_connection_factory(self) -> None:
        stub = _StubRun()
        original = fre._run_one  # noqa: SLF001
        fre._run_one = stub  # noqa: SLF001
        try:
            await fre.run_floor_reach_eval(
                character_id="IRONCLAD",
                num_runs=2,
                ports=[9, 10],
                connection_factory=lambda: object(),
            )
        finally:
            fre._run_one = original  # noqa: SLF001

        self.assertEqual(len(stub.ports), 2)
