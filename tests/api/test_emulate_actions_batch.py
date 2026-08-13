"""Coverage for `AsyncTrainingApiClient.emulate_actions` (DTO v0.8 batch operation).

The batch operation stays on the existing single in-flight request stream. Every item
parent must already exist when the request starts, exact replay applies to the entire
batch, audit identity is item-scoped, and response correlation is checked per item.
"""

from __future__ import annotations

import unittest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.contract import ApiProtocolError, MASK_VERSION, SCHEMA_VERSION
from sts2_training.api.transport import RetryRequest, TransportError


class _EmulateActionsConnection:
    client_session_id = "session-a"

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.fail_operation_once: str | None = None
        self.branch_results_transform = None

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
        operation = request["operation"]

        if operation == "start_instance":
            return {
                **self._common(request),
                "status": "completed",
                "instance_id": "inst-001",
                "max_emulate_actions_items": 64,
            }

        if operation == "emulate_actions":
            if self.fail_operation_once == operation:
                self.fail_operation_once = None
                raise TransportError(
                    "lost emulate_actions response",
                    completion_uncertain=True,
                    retry_request=RetryRequest.from_message(request),
                )
            branch_results = {}
            for item in request["items"]:
                branch_results[item["branch_id"]] = {
                    "status": "completed",
                    "branch_id": item["branch_id"],
                    "parent_branch_id": item["parent_branch_id"],
                    "rng_id": item["rng_id"],
                    "decision_point_id": f"d-{item['branch_id']}-001",
                    "branch_log": [],
                    "masked_emulator_dto": {
                        "mask_version": MASK_VERSION,
                        "legal_actions": [{"action_id": "a-001"}],
                    },
                }
            if self.branch_results_transform is not None:
                branch_results = self.branch_results_transform(request, branch_results)
            return {
                **self._common(request),
                "instance_id": request["instance_id"],
                "status": "completed",
                "branch_results": branch_results,
            }

        raise AssertionError(f"unexpected operation: {operation}")

    async def invalidate(self) -> None:
        pass

    async def close(self) -> None:
        pass


class EmulateActionsBatchTest(unittest.IsolatedAsyncioTestCase):
    async def _started_client(
        self, *, selection_logger=None
    ) -> tuple[AsyncTrainingApiClient, _EmulateActionsConnection, str]:
        connection = _EmulateActionsConnection()
        client = AsyncTrainingApiClient(  # type: ignore[arg-type]
            connection, selection_logger=selection_logger
        )
        instance_id = await client.start_instance({"instance_type": "combat"}, timeout_s=1.0)
        return client, connection, instance_id

    @staticmethod
    def _root_items() -> list[dict]:
        return [
            {
                "parent_branch_id": "root",
                "branch_id": "b1",
                "rng_id": 1,
                "decision_point_id": "d-root-001",
                "action_id": "a-001",
            },
            {
                "parent_branch_id": "root",
                "branch_id": "b2",
                "rng_id": 2,
                "decision_point_id": "d-root-001",
                "action_id": "a-001",
            },
        ]

    async def test_multi_parent_batch_sent_as_a_single_request(self) -> None:
        client, connection, instance_id = await self._started_client()
        try:
            # Same-batch newly-created parents are forbidden. Create b1 and b2 first,
            # then use both already-existing Branches as parents in the target batch.
            prepared = await client.emulate_actions(
                instance_id, self._root_items(), timeout_s=1.0
            )
            response = await client.emulate_actions(
                instance_id,
                [
                    {
                        "parent_branch_id": "b1",
                        "branch_id": "c1",
                        "rng_id": 1,
                        "decision_point_id": prepared["branch_results"]["b1"]["decision_point_id"],
                        "action_id": "a-001",
                    },
                    {
                        "parent_branch_id": "b2",
                        "branch_id": "c2",
                        "rng_id": 2,
                        "decision_point_id": prepared["branch_results"]["b2"]["decision_point_id"],
                        "action_id": "a-001",
                    },
                ],
                timeout_s=1.0,
            )
            self.assertEqual(response["status"], "completed")
            self.assertEqual(response["branch_results"]["c1"]["status"], "completed")
            self.assertEqual(response["branch_results"]["c2"]["status"], "completed")

            batch_requests = [m for m in connection.messages if m["operation"] == "emulate_actions"]
            self.assertEqual(len(batch_requests), 2)
            target = batch_requests[-1]
            self.assertEqual(len(target["items"]), 2)
            self.assertEqual(
                {item["parent_branch_id"] for item in target["items"]}, {"b1", "b2"}
            )
            self.assertEqual(client.next_request_seq, 4)
        finally:
            await client.close()

    async def test_selection_logger_records_each_batch_item_as_selection(self) -> None:
        events: list[dict] = []
        client, _, instance_id = await self._started_client(selection_logger=events.append)
        try:
            response = await client.emulate_actions(
                instance_id, self._root_items(), timeout_s=1.0
            )
            self.assertEqual(response["status"], "completed")
            self.assertEqual([event["event"] for event in events], ["selection", "selection"])
            self.assertEqual(
                [event["request"]["branch_id"] for event in events], ["b1", "b2"]
            )
            self.assertTrue(
                all(event["request"]["operation"] == "emulate_actions" for event in events)
            )
        finally:
            await client.close()

    async def test_retry_audit_is_item_scoped_and_does_not_duplicate_selection(self) -> None:
        events: list[dict] = []
        client, connection, instance_id = await self._started_client(
            selection_logger=events.append
        )
        try:
            items = self._root_items()
            connection.fail_operation_once = "emulate_actions"
            with self.assertRaisesRegex(TransportError, "lost emulate_actions response"):
                await client.emulate_actions(instance_id, items, timeout_s=1.0)

            retry = client.pending_retry
            self.assertIsNotNone(retry)
            assert retry is not None
            request_id = retry.request_id
            first_attempt_events = [
                event for event in events if event["request"]["request_id"] == request_id
            ]
            self.assertEqual(
                [event["event"] for event in first_attempt_events],
                ["selection", "selection"],
            )
            self.assertEqual(
                [event["request"]["branch_id"] for event in first_attempt_events],
                ["b1", "b2"],
            )
            self.assertTrue(all("client_error" in event for event in first_attempt_events))

            response = await client.retry_request(retry, timeout_s=1.0)
            self.assertEqual(response["status"], "completed")
            request_events = [
                event for event in events if event["request"]["request_id"] == request_id
            ]
            self.assertEqual(
                [event["event"] for event in request_events],
                ["selection", "selection", "selection_recovery", "selection_recovery"],
            )
            self.assertEqual(
                [event["request"]["branch_id"] for event in request_events],
                ["b1", "b2", "b1", "b2"],
            )
            self.assertEqual(
                sum(event["event"] == "selection" for event in request_events), 2
            )
        finally:
            await client.close()

    async def test_retry_after_lost_response_replays_exact_same_batch_request(self) -> None:
        client, connection, instance_id = await self._started_client()
        try:
            items = self._root_items()
            connection.fail_operation_once = "emulate_actions"
            with self.assertRaisesRegex(TransportError, "lost emulate_actions response"):
                await client.emulate_actions(instance_id, items, timeout_s=1.0)

            retry = client.pending_retry
            self.assertIsNotNone(retry)
            assert retry is not None
            self.assertEqual(retry.request_seq, 2)
            lost_attempt = connection.messages[-1]
            self.assertEqual(retry.to_message(), lost_attempt)
            with self.assertRaisesRegex(RuntimeError, "unresolved request"):
                await client.emulate_actions(instance_id, items, timeout_s=1.0)

            response = await client.retry_request(retry, timeout_s=1.0)
            self.assertEqual(response["status"], "completed")
            self.assertEqual(response["branch_results"]["b1"]["status"], "completed")
            self.assertEqual(response["branch_results"]["b2"]["status"], "completed")
            self.assertEqual(connection.messages[-1], lost_attempt)
            self.assertIsNone(client.pending_retry)
            self.assertEqual(client.next_request_seq, 3)

            batch_requests = [m for m in connection.messages if m["operation"] == "emulate_actions"]
            self.assertEqual(len(batch_requests), 2)
            self.assertEqual(batch_requests[0], batch_requests[1])
        finally:
            await client.close()

    async def _assert_bad_branch_results(self, transform, message_pattern: str) -> None:
        client, connection, instance_id = await self._started_client()
        connection.branch_results_transform = transform
        try:
            with self.assertRaisesRegex(ApiProtocolError, message_pattern):
                await client.emulate_actions(
                    instance_id,
                    [self._root_items()[0]],
                    timeout_s=1.0,
                )
            self.assertIsNotNone(client.pending_retry)
            assert client.pending_retry is not None
            self.assertEqual(client.pending_retry.request_seq, 2)
        finally:
            await client.close()

    async def test_wrong_branch_id_is_protocol_error_and_stays_pending(self) -> None:
        def transform(request, results):
            results["b1"]["branch_id"] = "wrong"
            return results

        await self._assert_bad_branch_results(transform, "branch_id")

    async def test_wrong_parent_branch_id_is_protocol_error_and_stays_pending(self) -> None:
        def transform(request, results):
            results["b1"]["parent_branch_id"] = "wrong-parent"
            return results

        await self._assert_bad_branch_results(transform, "parent_branch_id")

    async def test_wrong_rng_id_is_protocol_error_and_stays_pending(self) -> None:
        def transform(request, results):
            results["b1"]["rng_id"] = 999
            return results

        await self._assert_bad_branch_results(transform, "rng_id")

    async def test_missing_branch_result_is_protocol_error_and_stays_pending(self) -> None:
        def transform(request, results):
            results.pop("b1")
            return results

        await self._assert_bad_branch_results(transform, "branch_results")

    async def test_unexpected_extra_branch_result_is_protocol_error_and_stays_pending(self) -> None:
        def transform(request, results):
            results["extra"] = {
                "status": "faulted",
                "branch_id": "extra",
                "parent_branch_id": "root",
                "rng_id": 99,
                "error": "unexpected",
            }
            return results

        await self._assert_bad_branch_results(transform, "branch_results")

    async def test_running_branch_result_is_protocol_error_and_stays_pending(self) -> None:
        def transform(request, results):
            results["b1"]["status"] = "running"
            return results

        await self._assert_bad_branch_results(transform, "invalid branch status")

    async def test_empty_items_rejected_before_any_request_is_sent(self) -> None:
        client, connection, instance_id = await self._started_client()
        try:
            with self.assertRaises(ValueError):
                await client.emulate_actions(instance_id, [], timeout_s=1.0)
            batch_requests = [m for m in connection.messages if m["operation"] == "emulate_actions"]
            self.assertEqual(batch_requests, [])
            self.assertEqual(client.next_request_seq, 2)
        finally:
            await client.close()

    async def test_duplicate_branch_id_within_batch_rejected_client_side(self) -> None:
        client, connection, instance_id = await self._started_client()
        try:
            with self.assertRaises(ValueError):
                await client.emulate_actions(
                    instance_id,
                    [
                        self._root_items()[0],
                        {
                            **self._root_items()[1],
                            "branch_id": "b1",
                        },
                    ],
                    timeout_s=1.0,
                )
            batch_requests = [m for m in connection.messages if m["operation"] == "emulate_actions"]
            self.assertEqual(batch_requests, [])
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()
