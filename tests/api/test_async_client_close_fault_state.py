from __future__ import annotations

import unittest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.contract import RequestFaultedError
from sts2_training.api.transport import RetryRequest


class _CloseFaultConnection:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def exchange(self, message, *, deadline: float):
        request = dict(message)
        self.messages.append(request)
        if request["operation"] == "start_instance":
            return {
                "schema_version": request["schema_version"],
                "request_id": request["request_id"],
                "operation": "start_instance",
                "status": "completed",
                "instance_id": "inst-001",
                "decision_point_id": "decision-1",
                "masked_emulator_dto": {"state": "initial"},
            }
        if request["operation"] == "close_instance":
            return {
                "schema_version": request["schema_version"],
                "request_id": request["request_id"],
                "operation": "close_instance",
                "status": "faulted",
                "instance_id": request["instance_id"],
                "error": "close failed after partial cleanup",
                "fault_kind": "RuntimeError",
            }
        raise AssertionError(f"unexpected operation: {request['operation']}")

    async def invalidate(self) -> None:
        pass

    async def close(self) -> None:
        pass


class AsyncClientCloseFaultStateTest(unittest.IsolatedAsyncioTestCase):
    async def test_faulted_close_discards_quarantined_active_instance(self) -> None:
        connection = _CloseFaultConnection()
        client = AsyncTrainingApiClient(connection)
        instance_id = await client.start_instance(
            {"instance_type": "combat"},
            timeout_s=1.0,
        )

        with self.assertRaises(RequestFaultedError):
            await client.close_instance(instance_id, timeout_s=1.0)

        self.assertIsNone(client.instance_id)
        self.assertFalse(client.close_uncertain)
        self.assertIsNone(client.pending_retry)

        before = len(connection.messages)
        with self.assertRaisesRegex(RuntimeError, "no active instance"):
            await client.get_decision(instance_id, timeout_s=1.0)
        self.assertEqual(len(connection.messages), before)

    async def test_reconciliation_is_rejected_while_operation_lock_is_held(self) -> None:
        connection = _CloseFaultConnection()
        client = AsyncTrainingApiClient(connection)
        client._start_uncertain = True
        client._pending_retry = RetryRequest.from_message(
            {
                "schema_version": "0.5",
                "request_id": "start-pending",
                "operation": "start_instance",
                "instance_config": {"instance_type": "combat"},
            }
        )

        await client._operation_lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "operation is in flight"):
                client.reconcile_start_uncertainty(instance_id=None)
        finally:
            client._operation_lock.release()

        self.assertTrue(client.start_uncertain)
        self.assertIsNotNone(client.pending_retry)

    async def test_instance_scoped_request_requires_active_instance(self) -> None:
        connection = _CloseFaultConnection()
        client = AsyncTrainingApiClient(connection)

        with self.assertRaisesRegex(RuntimeError, "no active instance"):
            await client.get_decision("foreign-instance", timeout_s=1.0)

        self.assertEqual(connection.messages, [])


if __name__ == "__main__":
    unittest.main()
