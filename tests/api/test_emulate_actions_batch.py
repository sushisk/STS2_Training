"""Coverage for `AsyncTrainingApiClient.emulate_actions` (DTO v0.7 batch operation).

Focuses on the Training-side contract this feature must preserve per
`docs/STS2_next_implementation_plan.md`:

* one batch is sent as exactly one request over the existing single in-flight
  session-sequenced protocol (`_operation_lock`, `request_seq`, `pending_retry`,
  exact request replay) - never as several concurrent requests.
* a lost batch response is recovered by replaying the EXACT same request, and a
  successful replay must not cause the request to be sent to RL twice as far as
  sequencing is concerned (the transport-level "was it applied twice" question is out
  of scope here - RL's own batch tests cover admission idempotency).
* the response envelope validates `branch_results` against the request's items.
"""

from __future__ import annotations

import unittest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.contract import ApiProtocolError
from sts2_training.api.transport import RetryRequest, TransportError


class _EmulateActionsConnection:
    client_session_id = "session-a"

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.fail_operation_once: str | None = None
        self.branch_results_by_attempt: list[dict] | None = None
        self._attempt = 0

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
            return {
                **self._common(request),
                "status": "completed",
                "instance_id": "inst-001",
            }

        if operation == "emulate_actions":
            if self.fail_operation_once == operation:
                self.fail_operation_once = None
                raise TransportError(
                    "lost emulate_actions response",
                    completion_uncertain=True,
                    retry_request=RetryRequest.from_message(request),
                )
            self._attempt += 1
            branch_results = {}
            for item in request["items"]:
                branch_results[item["branch_id"]] = {
                    "status": "completed",
                    "branch_id": item["branch_id"],
                    "parent_branch_id": item["parent_branch_id"],
                    "rng_id": item["rng_id"],
                    "decision_point_id": f"d-{item['branch_id']}-001",
                    "branch_log": [],
                    "masked_emulator_dto": {"legal_actions": [{"action_id": "a-001"}]},
                }
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
    async def _started_client(self) -> tuple[AsyncTrainingApiClient, _EmulateActionsConnection, str]:
        connection = _EmulateActionsConnection()
        client = AsyncTrainingApiClient(connection)  # type: ignore[arg-type]
        instance_id = await client.start_instance({"instance_type": "combat"}, timeout_s=1.0)
        return client, connection, instance_id

    async def test_multi_parent_batch_sent_as_a_single_request(self) -> None:
        client, connection, instance_id = await self._started_client()
        try:
            response = await client.emulate_actions(
                instance_id,
                [
                    {
                        "parent_branch_id": "root",
                        "branch_id": "b1",
                        "rng_id": 1,
                        "decision_point_id": "d-root-001",
                        "action_id": "a-001",
                    },
                    {
                        "parent_branch_id": "b1",
                        "branch_id": "b1a",
                        "rng_id": 1,
                        "decision_point_id": "d-b1-001",
                        "action_id": "a-001",
                    },
                ],
                timeout_s=1.0,
            )
            self.assertEqual(response["status"], "completed")
            self.assertEqual(response["branch_results"]["b1"]["status"], "completed")
            self.assertEqual(response["branch_results"]["b1a"]["status"], "completed")
            # Exactly one wire request carried both items - not two separate requests.
            batch_requests = [m for m in connection.messages if m["operation"] == "emulate_actions"]
            self.assertEqual(len(batch_requests), 1)
            self.assertEqual(len(batch_requests[0]["items"]), 2)
            self.assertEqual(client.next_request_seq, 3)
        finally:
            await client.close()

    async def test_retry_after_lost_response_replays_exact_same_batch_request(self) -> None:
        client, connection, instance_id = await self._started_client()
        try:
            items = [
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
            connection.fail_operation_once = "emulate_actions"
            with self.assertRaisesRegex(TransportError, "lost emulate_actions response"):
                await client.emulate_actions(instance_id, items, timeout_s=1.0)

            retry = client.pending_retry
            self.assertIsNotNone(retry)
            assert retry is not None
            self.assertEqual(retry.request_seq, 2)
            lost_attempt = connection.messages[-1]
            self.assertEqual(retry.to_message(), lost_attempt)
            # No fresh request may be issued while a batch retry is pending - this is
            # the same single in-flight rule as every other operation.
            with self.assertRaisesRegex(RuntimeError, "unresolved request"):
                await client.emulate_actions(instance_id, items, timeout_s=1.0)

            response = await client.retry_request(retry, timeout_s=1.0)
            self.assertEqual(response["status"], "completed")
            self.assertEqual(response["branch_results"]["b1"]["status"], "completed")
            self.assertEqual(response["branch_results"]["b2"]["status"], "completed")
            # The replayed wire request is byte-identical to the one that was lost -
            # RL is expected to admission-dedupe on exact-request replay, not on
            # Training re-deriving a "new" request for the same logical batch.
            self.assertEqual(connection.messages[-1], lost_attempt)
            self.assertIsNone(client.pending_retry)
            self.assertEqual(client.next_request_seq, 3)

            # Only two `emulate_actions` messages ever crossed the wire: the lost
            # attempt and its exact replay - never a third, "double executed" copy.
            batch_requests = [m for m in connection.messages if m["operation"] == "emulate_actions"]
            self.assertEqual(len(batch_requests), 2)
            self.assertEqual(batch_requests[0], batch_requests[1])
        finally:
            await client.close()

    async def test_branch_results_key_mismatch_raises_protocol_error_and_stays_pending(self) -> None:
        client, connection, instance_id = await self._started_client()

        async def _bad_exchange(message: dict, *, deadline: float) -> dict:
            request = dict(message)
            if request["operation"] == "emulate_actions":
                return {
                    **connection._common(request),
                    "instance_id": request["instance_id"],
                    "status": "completed",
                    "branch_results": {"unexpected-branch": {"status": "completed"}},
                }
            return await _EmulateActionsConnection.exchange(connection, message, deadline=deadline)

        connection.exchange = _bad_exchange  # type: ignore[assignment]
        try:
            with self.assertRaisesRegex(ApiProtocolError, "branch_results"):
                await client.emulate_actions(
                    instance_id,
                    [
                        {
                            "parent_branch_id": "root",
                            "branch_id": "b1",
                            "rng_id": 1,
                            "decision_point_id": "d-root-001",
                            "action_id": "a-001",
                        }
                    ],
                    timeout_s=1.0,
                )
            self.assertTrue(client.pending_retry is not None)
            self.assertEqual(client.pending_retry.request_seq, 2)
        finally:
            await client.close()

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
                        {
                            "parent_branch_id": "root",
                            "branch_id": "dup",
                            "rng_id": 1,
                            "decision_point_id": "d-root-001",
                            "action_id": "a-001",
                        },
                        {
                            "parent_branch_id": "root",
                            "branch_id": "dup",
                            "rng_id": 2,
                            "decision_point_id": "d-root-001",
                            "action_id": "a-001",
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
