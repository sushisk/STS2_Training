from __future__ import annotations

import asyncio
import json
import unittest

from sts2_training.api.tcp_connection import TcpConnection
from sts2_training.api.transport import TransportError
from sts2_training.selection_log import SelectionAudit


class RetryResponseLimitAndAuditTest(unittest.IsolatedAsyncioTestCase):
    async def test_response_limit_can_be_raised_before_exact_retry(self) -> None:
        async def handle_client(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            try:
                line = await reader.readline()
                request = json.loads(line)
                response = {
                    "schema_version": request["schema_version"],
                    "request_id": request["request_id"],
                    "operation": request["operation"],
                    "status": "completed",
                    "instance_id": request["instance_id"],
                    "branch_id": "root",
                    "decision_point_id": "decision-2",
                    "masked_emulator_dto": {"payload": "x" * 1024},
                }
                writer.write(
                    json.dumps(response, separators=(",", ":")).encode("utf-8")
                    + b"\n"
                )
                await writer.drain()
            except (ConnectionError, OSError):
                pass
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except (ConnectionError, OSError):
                    pass

        server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        connection = TcpConnection(port=port, max_response_bytes=128)
        request = {
            "schema_version": "0.5",
            "request_id": "req-retry-size",
            "operation": "commit_action",
            "instance_id": "inst-001",
        }

        try:
            with self.assertRaisesRegex(TransportError, "max_response_bytes=128") as caught:
                await connection.exchange(request, timeout_s=1.0)

            error = caught.exception
            self.assertTrue(error.completion_uncertain)
            self.assertIsNotNone(error.retry_request)
            retry = error.retry_request
            assert retry is not None
            self.assertEqual(retry.to_message(), request)

            await connection.set_max_response_bytes(4096)
            self.assertEqual(connection.max_response_bytes, 4096)
            response = await connection.exchange(retry.to_message(), timeout_s=1.0)
            self.assertEqual(response["request_id"], "req-retry-size")
            self.assertEqual(
                response["masked_emulator_dto"]["payload"],
                "x" * 1024,
            )
        finally:
            await connection.close()
            server.close()
            await server.wait_closed()

    async def test_invalid_response_limit_update_is_rejected(self) -> None:
        connection = TcpConnection()
        with self.assertRaisesRegex(ValueError, "max_response_bytes must be positive"):
            await connection.set_max_response_bytes(0)
        await connection.close()


class SelectionAuditRetryTest(unittest.TestCase):
    def test_exact_retry_records_recovery_without_second_selection(self) -> None:
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

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["event"], "selection")
        self.assertEqual(events[0]["request"]["request_id"], "req-commit-1")
        self.assertEqual(events[0]["client_error"]["type"], "TransportError")
        self.assertEqual(events[1]["event"], "selection_recovery")
        self.assertEqual(events[1]["request"]["request_id"], "req-commit-1")
        self.assertEqual(events[1]["result"], replay_result)

        next_request = {
            **request,
            "request_id": "req-commit-2",
            "decision_point_id": "decision-2",
            "action_id": "action-2",
        }
        audit.record_action(
            next_request,
            source_branch_id="root",
            result=None,
            error=RuntimeError("stop"),
        )
        self.assertEqual(len(events), 3)
        self.assertEqual(events[2]["received"]["decision_point_id"], "decision-2")


if __name__ == "__main__":
    unittest.main()
