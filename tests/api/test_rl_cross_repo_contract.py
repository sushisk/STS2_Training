from __future__ import annotations

import asyncio
import sys
import types
import unittest
from unittest.mock import patch

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.tcp_connection import TcpConnection

try:
    from API.server import RLApiServer
    from API.tcp_server import AsyncioTcpServer
except ModuleNotFoundError:
    RLApiServer = None
    AsyncioTcpServer = None


class _FakeCombatInstance:
    created = 0
    commit_calls = 0

    def __init__(self, instance_id: str, instance_config: dict, **kwargs) -> None:
        type(self).created += 1
        self.instance_id = instance_id
        self.closed = False

    def start_instance_response(self) -> dict:
        return {
            "status": "completed",
            "instance_id": self.instance_id,
        }

    def get_decision(self, branch_id: str) -> dict:
        return {
            "status": "completed",
            "instance_id": self.instance_id,
            "branch_id": branch_id,
            "decision_point_id": "d-root-001",
            "masked_emulator_dto": {
                "legal_actions": [],
                "padding": "x" * 2048,
            },
        }

    def commit_action(self, decision_point_id: str, action_id: str) -> dict:
        type(self).commit_calls += 1
        return {
            "status": "completed",
            "instance_id": self.instance_id,
            "branch_id": "root",
            "decision_point_id": "d-root-after-commit",
            "masked_emulator_dto": {
                "legal_actions": [],
                "padding": "x" * 2048,
            },
        }

    def close(self) -> None:
        self.closed = True


@unittest.skipIf(
    RLApiServer is None or AsyncioTcpServer is None,
    "STS2_RL checkout is not available on PYTHONPATH",
)
class RealRlTcpContractTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        _FakeCombatInstance.created = 0
        _FakeCombatInstance.commit_calls = 0
        fake_module = types.ModuleType("API.instance_combat")
        fake_module.CombatInstance = _FakeCombatInstance
        self.module_patch = patch.dict(sys.modules, {"API.instance_combat": fake_module})
        self.module_patch.start()

        self.dispatcher = RLApiServer()
        self.server = AsyncioTcpServer(self.dispatcher.handle_request, port=0)
        await self.server.start()

    async def asyncTearDown(self) -> None:
        await self.server.close()
        self.dispatcher.close_all()
        self.module_patch.stop()

    async def test_recreated_clients_do_not_collide_on_start_request_id(self) -> None:
        first_connection = TcpConnection(port=self.server.bound_port)
        first_client = AsyncTrainingApiClient(first_connection)
        first_id = await first_client.start_instance(
            {"instance_type": "combat"},
            timeout_s=1.0,
        )
        await first_client.close_instance(first_id, timeout_s=1.0)
        await first_client.close()

        second_connection = TcpConnection(port=self.server.bound_port)
        second_client = AsyncTrainingApiClient(second_connection)
        try:
            second_id = await second_client.start_instance(
                {"instance_type": "combat"},
                timeout_s=1.0,
            )
        finally:
            await second_client.close()

        self.assertNotEqual(second_id, first_id)
        self.assertEqual(_FakeCombatInstance.created, 2)

    async def test_same_request_retries_on_fresh_connection_are_replayed(self) -> None:
        request = {
            "schema_version": "0.5",
            "request_id": "req-cross-repo-retry",
            "operation": "start_instance",
            "instance_config": {"instance_type": "combat"},
        }
        connection = TcpConnection(port=self.server.bound_port)
        try:
            first = await connection.exchange(request, timeout_s=1.0)
            await connection.invalidate()
            replay = await connection.exchange(request, timeout_s=1.0)
        finally:
            await connection.close()

        self.assertEqual(replay, first)
        self.assertEqual(_FakeCombatInstance.created, 1)

    async def test_lost_close_response_can_be_retried_with_same_request_id(self) -> None:
        client = AsyncTrainingApiClient(TcpConnection(port=self.server.bound_port))
        instance_id = await client.start_instance(
            {"instance_type": "combat"},
            timeout_s=1.0,
        )
        await client.close()

        close_request = {
            "schema_version": "0.5",
            "request_id": "req-cross-repo-close",
            "operation": "close_instance",
            "instance_id": instance_id,
        }

        reader, writer = await asyncio.open_connection(
            "127.0.0.1",
            self.server.bound_port,
        )
        del reader
        writer.write(
            (
                '{"schema_version":"0.5","request_id":"req-cross-repo-close",'
                f'"operation":"close_instance","instance_id":"{instance_id}"}}\n'
            ).encode("utf-8")
        )
        await writer.drain()
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass

        retry_connection = TcpConnection(port=self.server.bound_port)
        try:
            response = await retry_connection.exchange(close_request, timeout_s=1.0)
        finally:
            await retry_connection.close()

        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["request_id"], close_request["request_id"])
        self.assertEqual(self.dispatcher.instance_count(), 0)

    async def test_large_non_idempotent_response_is_delivered_and_replayed(self) -> None:
        await self.server.close()
        self.server = AsyncioTcpServer(
            self.dispatcher.handle_request,
            port=0,
            max_message_bytes=512,
        )
        await self.server.start()

        connection = TcpConnection(
            port=self.server.bound_port,
            max_message_bytes=512,
        )
        try:
            start_request = {
                "schema_version": "0.5",
                "request_id": "req-large-start",
                "operation": "start_instance",
                "instance_config": {"instance_type": "combat"},
            }
            started = await connection.exchange(start_request, timeout_s=1.0)
            commit_request = {
                "schema_version": "0.5",
                "request_id": "req-large-commit",
                "operation": "commit_action",
                "instance_id": started["instance_id"],
                "branch_id": "root",
                "rng_id": 0,
                "decision_point_id": "d-root-001",
                "action_id": "a-001",
            }

            first = await connection.exchange(commit_request, timeout_s=1.0)
            self.assertGreater(len(first["masked_emulator_dto"]["padding"]), 512)

            await connection.invalidate()
            replay = await connection.exchange(commit_request, timeout_s=1.0)
        finally:
            await connection.close()

        self.assertEqual(replay, first)
        self.assertEqual(_FakeCombatInstance.commit_calls, 1)


if __name__ == "__main__":
    unittest.main()
