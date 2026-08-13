"""Regression coverage for Training-vs-RL batch deadline semantics."""

from __future__ import annotations

import asyncio
import unittest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.contract import MASK_VERSION, SCHEMA_VERSION
from sts2_training.api.transport import RetryRequest, TransportError


class _TwoWorkerFourItemConnection:
    """Model a normal 2-worker/4-item batch whose wall clock exceeds client timeout."""

    client_session_id = "session-deadline"

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self._expire_first_batch = True

    @staticmethod
    def _common(request: dict) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "server_epoch": "epoch-1",
            "client_session_id": request["client_session_id"],
            "request_seq": request["request_seq"],
            "request_id": request["request_id"],
            "operation": request["operation"],
        }

    async def exchange(self, message: dict, *, deadline: float) -> dict:
        request = dict(message)
        self.messages.append(request)

        if request["operation"] == "start_instance":
            return {
                **self._common(request),
                "status": "completed",
                "instance_id": "inst-001",
                "max_emulate_actions_items": 64,
            }

        if request["operation"] != "emulate_actions":
            raise AssertionError(f"unexpected operation: {request['operation']}")

        assert len(request["items"]) == 4  # more items than the modeled 2 workers
        assert request["simulation_options"]["max_time_ms"] == 60_000

        if self._expire_first_batch:
            self._expire_first_batch = False
            # TcpConnection owns the end-to-end client deadline. Model a request that
            # remains valid on RL (per-Branch max_time_ms has not expired) while the
            # shorter Training operation deadline expires first.
            delay = max(0.0, deadline - asyncio.get_running_loop().time())
            await asyncio.sleep(delay)
            raise TransportError(
                "Training operation deadline expired before batch response",
                completion_uncertain=True,
                retry_request=RetryRequest.from_message(request),
            )

        branch_results = {
            item["branch_id"]: {
                "status": "completed",
                "branch_id": item["branch_id"],
                "parent_branch_id": item["parent_branch_id"],
                "rng_id": item["rng_id"],
                "decision_point_id": f"d-{item['branch_id']}-next",
                "branch_log": [],
                "masked_emulator_dto": {
                    "mask_version": MASK_VERSION,
                    "legal_actions": [{"action_id": "a-next"}],
                },
            }
            for item in request["items"]
        }
        return {
            **self._common(request),
            "instance_id": request["instance_id"],
            "status": "completed",
            "branch_results": branch_results,
        }

    async def invalidate(self) -> None:
        pass

    async def close(self) -> None:
        pass


class EmulateActionsDeadlineSemanticsTest(unittest.IsolatedAsyncioTestCase):
    async def test_client_deadline_can_expire_before_per_branch_timeout_and_exact_replay(self) -> None:
        connection = _TwoWorkerFourItemConnection()
        client = AsyncTrainingApiClient(connection)  # type: ignore[arg-type]
        instance_id = await client.start_instance({"instance_type": "combat"}, timeout_s=1.0)
        items = [
            {
                "parent_branch_id": "root",
                "branch_id": f"b{index}",
                "rng_id": index + 1,
                "decision_point_id": "d-root-001",
                "action_id": "a-001",
            }
            for index in range(4)
        ]

        try:
            with self.assertRaisesRegex(TransportError, "operation deadline expired"):
                await client.emulate_actions(
                    instance_id,
                    items,
                    timeout_s=0.01,
                    simulation_options={"max_time_ms": 60_000},
                )

            pending = client.pending_retry
            self.assertIsNotNone(pending)
            assert pending is not None
            first_request = connection.messages[-1]
            self.assertEqual(pending.to_message(), first_request)
            self.assertEqual(first_request["simulation_options"]["max_time_ms"], 60_000)
            self.assertEqual(client.next_request_seq, 2)

            replayed = await client.retry_request(pending, timeout_s=1.0)
            self.assertEqual(replayed["status"], "completed")
            self.assertEqual(set(replayed["branch_results"]), {"b0", "b1", "b2", "b3"})
            self.assertEqual(connection.messages[-1], first_request)
            self.assertIsNone(client.pending_retry)
            self.assertEqual(client.next_request_seq, 3)
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()
