from __future__ import annotations

import unittest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.contract import ApiProtocolError
from sts2_training.api.transport import TransportError
from sts2_training.selection_log import SelectionAudit


class _MalformedCommitConnection:
    async def exchange(self, message, *, deadline: float):
        request = dict(message)
        operation = request["operation"]
        common = {
            "schema_version": request["schema_version"],
            "request_id": request["request_id"],
            "operation": operation,
        }
        if operation == "start_instance":
            return {
                **common,
                "status": "completed",
                "instance_id": "inst-001",
            }
        if operation == "commit_action":
            return {
                **common,
                "status": "completed",
                "instance_id": request["instance_id"],
                "branch_id": "root",
                "decision_point_id": "decision-2",
                # Correlated envelope, but deliberately malformed operation payload.
            }
        if operation == "get_decision":
            return {
                **common,
                "status": "completed",
                "instance_id": request["instance_id"],
                "branch_id": request["branch_id"],
                "decision_point_id": "decision-reconciled",
                "masked_emulator_dto": {"state": "reconciled"},
            }
        raise AssertionError(f"unexpected operation: {operation}")

    async def invalidate(self) -> None:
        pass

    async def close(self) -> None:
        pass


class RecoveryRegressionTest(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_action_replay_can_be_explicitly_reconciled(self) -> None:
        client = AsyncTrainingApiClient(_MalformedCommitConnection())
        instance_id = await client.start_instance(
            {"instance_type": "combat"},
            timeout_s=1.0,
        )

        with self.assertRaisesRegex(ApiProtocolError, "masked_emulator_dto"):
            await client.commit_action(
                instance_id,
                "decision-1",
                "action-1",
                timeout_s=1.0,
            )

        retry = client.pending_retry
        self.assertIsNotNone(retry)
        assert retry is not None

        with self.assertRaisesRegex(ApiProtocolError, "masked_emulator_dto"):
            await client.retry_request(retry, timeout_s=1.0)
        self.assertEqual(client.pending_retry, retry)

        client.reconcile_pending_uncertainty()
        self.assertIsNone(client.pending_retry)

        decision = await client.get_decision(instance_id, timeout_s=1.0)
        self.assertEqual(decision["decision_point_id"], "decision-reconciled")


class SelectionAuditRecoveryTest(unittest.TestCase):
    def test_same_id_retry_records_final_outcome_without_second_selection(self) -> None:
        events: list[dict] = []
        audit = SelectionAudit(events.append)
        audit.remember(
            {
                "instance_id": "inst-001",
                "branch_id": "root",
                "decision_point_id": "decision-1",
                "masked_emulator_dto": {"state": "before"},
            }
        )
        request = {
            "schema_version": "0.5",
            "request_id": "req-commit-1",
            "operation": "commit_action",
            "instance_id": "inst-001",
            "branch_id": "root",
            "decision_point_id": "decision-1",
            "action_id": "action-1",
        }

        audit.record_action(
            request,
            source_branch_id="root",
            result=None,
            error=TransportError("lost response", completion_uncertain=True),
        )
        replay_result = {
            "status": "completed",
            "instance_id": "inst-001",
            "branch_id": "root",
            "decision_point_id": "decision-2",
            "masked_emulator_dto": {"state": "after"},
        }
        audit.record_action(
            request,
            source_branch_id="root",
            result=replay_result,
        )

        self.assertEqual([event["event"] for event in events], [
            "selection",
            "selection_recovery",
        ])
        self.assertEqual(events[0]["client_error"]["type"], "TransportError")
        self.assertEqual(events[1]["result"], replay_result)
        self.assertEqual(
            sum(event["event"] == "selection" for event in events),
            1,
        )


if __name__ == "__main__":
    unittest.main()
