from __future__ import annotations

import unittest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.contract import RequestFaultedError, SCHEMA_VERSION


def _common(request: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "server_epoch": "epoch-1",
        "client_session_id": request["client_session_id"],
        "request_seq": request["request_seq"],
        "request_id": request["request_id"],
        "operation": request["operation"],
    }


class _FaultedCloseConnection:
    client_session_id = "session-a"

    async def exchange(self, message: dict, *, deadline: float) -> dict:
        request = dict(message)
        if request["operation"] == "start_instance":
            instance_type = request["instance_config"]["instance_type"]
            response = {
                **_common(request),
                "status": "completed",
                "instance_id": f"inst-{request['request_seq']}",
            }
            if instance_type == "combat":
                response["max_emulate_actions_items"] = 64
            return response
        if request["operation"] == "close_instance":
            return {
                **_common(request),
                "status": "faulted",
                "instance_id": request["instance_id"],
                "error": "close failed after the instance became unusable",
                "fault_kind": "emulator_error",
            }
        raise AssertionError(f"unexpected operation {request['operation']!r}")


class InstanceTypeLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def test_faulted_close_clears_active_instance_type(self) -> None:
        client = AsyncTrainingApiClient(_FaultedCloseConnection())  # type: ignore[arg-type]
        instance_id = await client.start_instance(
            {"instance_type": "whole_run"},
            timeout_s=1.0,
        )

        self.assertEqual(client.instance_id, instance_id)
        self.assertEqual(client.instance_type, "whole_run")

        with self.assertRaises(RequestFaultedError):
            await client.close_instance(instance_id, timeout_s=1.0)

        self.assertIsNone(client.instance_id)
        self.assertIsNone(client.instance_type)
        self.assertIsNone(client.max_emulate_actions_items)

        next_instance_id = await client.start_instance(
            {"instance_type": "combat"},
            timeout_s=1.0,
        )
        self.assertEqual(client.instance_id, next_instance_id)
        self.assertEqual(client.instance_type, "combat")
        self.assertEqual(client.max_emulate_actions_items, 64)


if __name__ == "__main__":
    unittest.main()
