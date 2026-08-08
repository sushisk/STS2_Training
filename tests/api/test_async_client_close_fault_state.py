from __future__ import annotations

import unittest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.contract import RequestFaultedError


class _CloseFaultConnection:
    client_session_id = "session-a"

    def __init__(self) -> None:
        self.messages: list[dict] = []

    @staticmethod
    def _common(request: dict) -> dict:
        return {
            "schema_version": "0.7",
            "server_epoch": "epoch-1",
            "client_session_id": request["client_session_id"],
            "request_seq": request["request_seq"],
            "request_id": request["request_id"],
            "operation": request["operation"],
        }

    async def exchange(self, message, *, deadline: float):
        request = dict(message)
        self.messages.append(request)
        if request["operation"] == "start_instance":
            return {
                **self._common(request),
                "status": "completed",
                "instance_id": "inst-001",
                "max_emulate_actions_items": 64,
                "decision_point_id": "decision-1",
                "masked_emulator_dto": {"state": "initial"},
            }
        if request["operation"] == "close_instance":
            return {
                **self._common(request),
                "status": "faulted",
                "instance_id": request["instance_id"],
                "error": "close failed after partial cleanup",
                "fault_kind": "emulator_error",
            }
        raise AssertionError(f"unexpected operation: {request['operation']}")

    async def invalidate(self) -> None:
        pass

    async def close(self) -> None:
        pass


class AsyncClientCloseFaultStateTest(unittest.IsolatedAsyncioTestCase):
    async def test_faulted_close_consumes_sequence_and_discards_quarantined_instance(self) -> None:
        connection = _CloseFaultConnection()
        client = AsyncTrainingApiClient(connection)
        instance_id = await client.start_instance(
            {"instance_type": "combat"}, timeout_s=1.0
        )
        self.assertEqual(client.max_emulate_actions_items, 64)

        with self.assertRaises(RequestFaultedError):
            await client.close_instance(instance_id, timeout_s=1.0)

        self.assertIsNone(client.instance_id)
        self.assertIsNone(client.max_emulate_actions_items)
        self.assertFalse(client.close_uncertain)
        self.assertIsNone(client.pending_retry)
        self.assertEqual(client.next_request_seq, 3)

        before = len(connection.messages)
        with self.assertRaisesRegex(RuntimeError, "no active instance"):
            await client.get_decision(instance_id, timeout_s=1.0)
        self.assertEqual(len(connection.messages), before)

    async def test_instance_scoped_request_requires_active_instance_before_send(self) -> None:
        connection = _CloseFaultConnection()
        client = AsyncTrainingApiClient(connection)

        with self.assertRaisesRegex(RuntimeError, "no active instance"):
            await client.get_decision("foreign-instance", timeout_s=1.0)

        self.assertEqual(connection.messages, [])
        self.assertEqual(client.next_request_seq, 1)


if __name__ == "__main__":
    unittest.main()
