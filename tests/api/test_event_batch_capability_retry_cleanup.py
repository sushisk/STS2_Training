from __future__ import annotations

import unittest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.contract import RequestFaultedError
from sts2_training.api.transport import RetryRequest, TransportError


class _Connection:
    client_session_id = "session-event-cleanup"

    def __init__(self, *, advertise_event_batch: bool = True) -> None:
        self.advertise_event_batch = advertise_event_batch
        self.messages: list[dict] = []
        self.fail_emulate_once = False
        self.fault_emulate_after_retry = False

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

    async def exchange(self, message: dict, *, deadline: float) -> dict:
        request = dict(message)
        self.messages.append(request)
        operation = request["operation"]

        if operation == "start_instance":
            response = {
                **self._common(request),
                "status": "completed",
                "instance_id": "inst-001",
                "max_emulate_actions_items": 64,
            }
            if self.advertise_event_batch:
                response["emulate_actions_boundaries"] = [
                    "event_choice",
                    "pending_choice",
                    "stable",
                ]
            return response

        if operation == "emulate_actions":
            if self.fail_emulate_once:
                self.fail_emulate_once = False
                raise TransportError(
                    "lost event batch response",
                    completion_uncertain=True,
                    retry_request=RetryRequest.from_message(request),
                )
            if self.fault_emulate_after_retry:
                return {
                    **self._common(request),
                    "status": "faulted",
                    "instance_id": request["instance_id"],
                    "error": "synthetic definitive fault",
                    "fault_kind": "synthetic_fault",
                }
            return {
                **self._common(request),
                "status": "completed",
                "instance_id": request["instance_id"],
                "branch_results": {
                    item["branch_id"]: {
                        "status": "completed",
                        "branch_id": item["branch_id"],
                        "parent_branch_id": item["parent_branch_id"],
                        "rng_id": item["rng_id"],
                        "decision_point_id": f"d-{item['branch_id']}-001",
                        "branch_log": [],
                        "masked_emulator_dto": {
                            "legal_actions": [{"action_id": "next"}],
                            "hp": 50,
                        },
                    }
                    for item in request["items"]
                },
            }

        if operation == "release_branches":
            return {
                **self._common(request),
                "status": "completed",
                "instance_id": request["instance_id"],
                "branch_statuses": {
                    branch_id: "released" for branch_id in request["branch_ids"]
                },
            }

        raise AssertionError(f"unexpected operation: {operation}")

    async def invalidate(self) -> None:
        pass

    async def close(self) -> None:
        pass


class EventBatchCapabilityRetryCleanupTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _items() -> list[dict]:
        return [
            {
                "parent_branch_id": "root",
                "branch_id": "event-b1",
                "rng_id": 1,
                "decision_point_id": "d-root-001",
                "action_id": "a-1",
            },
            {
                "parent_branch_id": "root",
                "branch_id": "event-b2",
                "rng_id": 2,
                "decision_point_id": "d-root-001",
                "action_id": "a-2",
            },
        ]

    async def test_start_instance_records_semantic_batch_capability(self) -> None:
        connection = _Connection(advertise_event_batch=True)
        client = AsyncTrainingApiClient(connection)  # type: ignore[arg-type]
        try:
            await client.start_instance({"instance_type": "whole_run"}, timeout_s=1.0)
            self.assertIn("event_choice", client.emulate_actions_boundaries)
        finally:
            await client.close()

        legacy_connection = _Connection(advertise_event_batch=False)
        legacy_client = AsyncTrainingApiClient(legacy_connection)  # type: ignore[arg-type]
        try:
            await legacy_client.start_instance(
                {"instance_type": "whole_run"}, timeout_s=1.0
            )
            self.assertEqual(legacy_client.emulate_actions_boundaries, frozenset())
        finally:
            await legacy_client.close()

    async def test_exact_retry_releases_owned_branches_before_returning(self) -> None:
        connection = _Connection()
        client = AsyncTrainingApiClient(connection)  # type: ignore[arg-type]
        try:
            instance_id = await client.start_instance(
                {"instance_type": "whole_run"}, timeout_s=1.0
            )
            connection.fail_emulate_once = True
            items = self._items()

            with self.assertRaisesRegex(TransportError, "lost event batch response"):
                await client.emulate_actions(instance_id, items, timeout_s=1.0)

            retry = client.pending_retry
            self.assertIsNotNone(retry)
            assert retry is not None
            client.defer_branch_cleanup_after_retry(
                retry,
                instance_id,
                [item["branch_id"] for item in items],
            )

            recovered = await client.retry_request(retry, timeout_s=1.0)
            self.assertEqual(recovered["status"], "completed")
            self.assertIsNone(client.pending_retry)
            self.assertEqual(client.next_request_seq, 4)
            self.assertEqual(
                [message["operation"] for message in connection.messages],
                [
                    "start_instance",
                    "emulate_actions",
                    "emulate_actions",
                    "release_branches",
                ],
            )
            self.assertEqual(
                connection.messages[-1]["branch_ids"],
                ["event-b1", "event-b2"],
            )
        finally:
            await client.close()

    async def test_definitive_retry_fault_still_releases_owned_branches(self) -> None:
        connection = _Connection()
        client = AsyncTrainingApiClient(connection)  # type: ignore[arg-type]
        try:
            instance_id = await client.start_instance(
                {"instance_type": "whole_run"}, timeout_s=1.0
            )
            connection.fail_emulate_once = True
            connection.fault_emulate_after_retry = True
            items = self._items()

            with self.assertRaises(TransportError):
                await client.emulate_actions(instance_id, items, timeout_s=1.0)
            retry = client.pending_retry
            assert retry is not None
            client.defer_branch_cleanup_after_retry(
                retry,
                instance_id,
                [item["branch_id"] for item in items],
            )

            with self.assertRaisesRegex(RequestFaultedError, "synthetic definitive fault"):
                await client.retry_request(retry, timeout_s=1.0)

            self.assertIsNone(client.pending_retry)
            self.assertEqual(client.next_request_seq, 4)
            self.assertEqual(
                [message["operation"] for message in connection.messages],
                [
                    "start_instance",
                    "emulate_actions",
                    "emulate_actions",
                    "release_branches",
                ],
            )
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()
