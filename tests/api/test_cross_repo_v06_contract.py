from __future__ import annotations

import asyncio
import importlib
import os
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.tcp_connection import TcpConnection
from sts2_training.api.transport import TransportError


def _find_rl_root() -> Path | None:
    configured = os.environ.get("STS2_RL_ROOT")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(Path(__file__).resolve().parents[3] / "STS2_RL")
    for candidate in candidates:
        if (candidate / "API" / "server.py").is_file() and (
            candidate / "API" / "tcp_server.py"
        ).is_file():
            return candidate.resolve()
    return None


class _FakeCombatInstance:
    creations = 0
    start_entered = threading.Event()
    release_start = threading.Event()

    def __init__(self, instance_id: str, instance_config: dict, **kwargs) -> None:
        type(self).creations += 1
        self.instance_id = instance_id

    def start_instance_response(self) -> dict:
        type(self).start_entered.set()
        if not type(self).release_start.wait(timeout=2.0):
            raise RuntimeError("test did not release start_instance_response")
        return {"status": "completed", "instance_id": self.instance_id}

    def get_decision(self, branch_id: str) -> dict:
        return {
            "status": "completed",
            "branch_id": branch_id,
            "decision_point_id": "decision-1",
            "masked_emulator_dto": {"state": "cross-repo"},
        }

    def close(self) -> None:
        return None


class CrossRepoV06ContractTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rl_root = _find_rl_root()
        if cls.rl_root is None:
            raise unittest.SkipTest(
                "set STS2_RL_ROOT or place STS2_RL beside STS2_Training to run cross-repo tests"
            )
        cls._preexisting_api_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "API" or name.startswith("API.")
        }
        cls._inserted_path = str(cls.rl_root)
        sys.path.insert(0, cls._inserted_path)
        cls.rl_server_module = importlib.import_module("API.server")
        cls.rl_tcp_module = importlib.import_module("API.tcp_server")

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "_inserted_path", None) in sys.path:
            sys.path.remove(cls._inserted_path)
        for name in list(sys.modules):
            if name == "API" or name.startswith("API."):
                sys.modules.pop(name, None)
        for name, module in getattr(cls, "_preexisting_api_modules", {}).items():
            sys.modules[name] = module

    async def asyncSetUp(self) -> None:
        _FakeCombatInstance.creations = 0
        _FakeCombatInstance.start_entered = threading.Event()
        _FakeCombatInstance.release_start = threading.Event()

        fake_module = types.ModuleType("API.instance_combat")
        fake_module.CombatInstance = _FakeCombatInstance
        self.module_patch = patch.dict(sys.modules, {"API.instance_combat": fake_module})
        self.module_patch.start()

        self.dispatcher = self.rl_server_module.RLApiServer(server_epoch="epoch-cross-repo")
        self.server = self.rl_tcp_module.AsyncioTcpServer(
            self.dispatcher.handle_request,
            server_epoch=self.dispatcher.server_epoch,
            port=0,
        )
        await self.server.start()

        self.connection = TcpConnection(
            host="127.0.0.1",
            port=self.server.bound_port,
            client_session_id="cross-repo-session",
            connect_timeout_s=1.0,
        )
        self.client = AsyncTrainingApiClient(self.connection)

    async def asyncTearDown(self) -> None:
        _FakeCombatInstance.release_start.set()
        await self.client.close()
        await self.server.close()
        self.dispatcher.close_all()
        self.module_patch.stop()

    async def test_real_rl_and_training_heads_complete_basic_lifecycle(self) -> None:
        _FakeCombatInstance.release_start.set()
        instance_id = await self.client.start_instance(
            {"instance_type": "combat"}, timeout_s=1.0
        )
        decision = await self.client.get_decision(instance_id, timeout_s=1.0)
        closed = await self.client.close_instance(instance_id, timeout_s=1.0)

        self.assertEqual(decision["decision_point_id"], "decision-1")
        self.assertEqual(decision["masked_emulator_dto"], {"state": "cross-repo"})
        self.assertEqual(closed["status"], "completed")
        self.assertEqual(self.connection.server_epoch, "epoch-cross-repo")

    async def test_timeout_retries_exact_request_without_duplicate_start(self) -> None:
        first_attempt = asyncio.create_task(
            self.client.start_instance({"instance_type": "combat"}, timeout_s=0.1)
        )
        entered = await asyncio.to_thread(_FakeCombatInstance.start_entered.wait, 0.5)
        self.assertTrue(entered, "RL did not begin start_instance before timeout")

        with self.assertRaises(TransportError):
            await first_attempt

        retry = self.client.pending_retry
        self.assertIsNotNone(retry)
        assert retry is not None

        _FakeCombatInstance.release_start.set()
        instance_id = await self.client.retry_request(retry, timeout_s=1.0)

        self.assertIsInstance(instance_id, str)
        self.assertTrue(instance_id)
        self.assertEqual(_FakeCombatInstance.creations, 1)
        self.assertIsNone(self.client.pending_retry)
        self.assertEqual(self.client.next_request_seq, 2)


if __name__ == "__main__":
    unittest.main()
