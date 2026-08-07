from __future__ import annotations

import asyncio
import json
import unittest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.contract import ApiProtocolError, RequestRejectedError
from sts2_training.api.tcp_connection import TcpConnection
from sts2_training.api.transport import TransportError


class AsyncTrainingApiClientTcpTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.requests: list[dict] = []
        self.connection_count = 0
        self.server = await asyncio.start_server(self._handle_client, "127.0.0.1", 0)
        self.port = int(self.server.sockets[0].getsockname()[1])
        self.connection = TcpConnection(
            port=self.port,
            client_session_id="session-a",
        )
        self.client = AsyncTrainingApiClient(self.connection)

    async def asyncTearDown(self) -> None:
        await self.client.close()
        self.server.close()
        await self.server.wait_closed()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.connection_count += 1
        try:
            while line := await reader.readline():
                request = json.loads(line)
                if request.get("transport_operation") == "hello":
                    response = {
                        "transport_operation": "hello",
                        "schema_version": "0.6",
                        "client_session_id": request["client_session_id"],
                        "server_epoch": "epoch-1",
                    }
                elif request == {"transport_operation": "ping"}:
                    response = {
                        "transport_operation": "pong",
                        "server_epoch": "epoch-1",
                    }
                else:
                    self.requests.append(request)
                    response = self._response_for(request)
                writer.write(json.dumps(response).encode("utf-8") + b"\n")
                await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    @staticmethod
    def _common(request: dict) -> dict:
        return {
            "schema_version": "0.6",
            "server_epoch": "epoch-1",
            "client_session_id": request["client_session_id"],
            "request_seq": request["request_seq"],
            "request_id": request["request_id"],
            "operation": request["operation"],
        }

    @classmethod
    def _response_for(cls, request: dict) -> dict:
        common = cls._common(request)
        operation = request["operation"]
        if operation == "start_instance":
            instance_type = request["instance_config"].get("instance_type")
            if instance_type == "reject":
                return {**common, "status": "rejected", "error": "bad config"}
            if instance_type == "mismatch":
                return {
                    **common,
                    "request_id": "wrong-request-id",
                    "status": "completed",
                    "instance_id": "inst-001",
                }
            if instance_type == "missing-instance":
                return {**common, "status": "completed"}
            return {
                **common,
                "status": "completed",
                "instance_id": "inst-001",
            }

        response = {
            **common,
            "status": "completed",
            "instance_id": request["instance_id"],
        }
        if operation == "get_decision":
            response.update(
                branch_id=request["branch_id"],
                decision_point_id="d-root-001",
                masked_emulator_dto={"legal_actions": [{"action_id": "a-001"}]},
            )
        elif operation == "commit_action":
            response.update(
                branch_id="root",
                decision_point_id="d-root-002",
                masked_emulator_dto={"legal_actions": []},
            )
        elif operation == "emulate_action":
            response.update(
                branch_id=request["branch_id"],
                parent_branch_id=request["parent_branch_id"],
                rng_id=request["rng_id"],
                decision_point_id="d-branch-001",
                masked_emulator_dto={"legal_actions": []},
            )
        return response

    async def test_start_instance_sends_v06_session_sequence(self) -> None:
        instance_id = await self.client.start_instance(
            {"instance_type": "combat"}, timeout_s=1.0
        )

        self.assertEqual(instance_id, "inst-001")
        self.assertEqual(
            self.requests,
            [
                {
                    "schema_version": "0.6",
                    "client_session_id": "session-a",
                    "request_seq": 1,
                    "request_id": "session-a:1",
                    "operation": "start_instance",
                    "instance_config": {"instance_type": "combat"},
                }
            ],
        )
        self.assertEqual(self.client.next_request_seq, 2)
        self.assertEqual(self.connection.server_epoch, "epoch-1")

    async def test_client_lock_wait_consumes_method_timeout_without_request(self) -> None:
        await self.client._operation_lock.acquire()
        try:
            with self.assertRaisesRegex(TransportError, "timed out before request"):
                await self.client.start_instance(
                    {"instance_type": "combat"}, timeout_s=0.01
                )
        finally:
            self.client._operation_lock.release()
        self.assertEqual(self.requests, [])
        self.assertEqual(self.client.next_request_seq, 1)

    async def test_concurrent_start_instance_creates_only_one_request(self) -> None:
        results = await asyncio.gather(
            self.client.start_instance({"instance_type": "combat-a"}, timeout_s=1.0),
            self.client.start_instance({"instance_type": "combat-b"}, timeout_s=1.0),
            return_exceptions=True,
        )
        successes = [result for result in results if isinstance(result, str)]
        self.assertEqual(successes, ["inst-001"])
        errors = [result for result in results if isinstance(result, RuntimeError)]
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.client.next_request_seq, 2)

    async def test_close_then_start_continues_same_session_sequence(self) -> None:
        instance_id = await self.client.start_instance(
            {"instance_type": "combat"}, timeout_s=1.0
        )
        await self.client.close_instance(instance_id, timeout_s=1.0)
        restarted = await self.client.start_instance(
            {"instance_type": "combat"}, timeout_s=1.0
        )

        self.assertEqual(restarted, "inst-001")
        self.assertEqual(
            [request["request_seq"] for request in self.requests], [1, 2, 3]
        )
        self.assertEqual(
            [request["operation"] for request in self.requests],
            ["start_instance", "close_instance", "start_instance"],
        )

    async def test_decision_and_commit_round_trip_over_tcp(self) -> None:
        instance_id = await self.client.start_instance(
            {"instance_type": "combat"}, timeout_s=1.0
        )
        decision = await self.client.get_decision(instance_id, timeout_s=1.0)
        committed = await self.client.commit_action(
            instance_id,
            decision["decision_point_id"],
            "a-001",
            timeout_s=1.0,
        )

        self.assertEqual(decision["branch_id"], "root")
        self.assertEqual(committed["decision_point_id"], "d-root-002")
        self.assertEqual(self.requests[-1]["request_seq"], 3)
        self.assertEqual(self.requests[-1]["request_id"], "session-a:3")

    async def test_emulate_action_round_trip_over_tcp(self) -> None:
        instance_id = await self.client.start_instance(
            {"instance_type": "combat"}, timeout_s=1.0
        )
        response = await self.client.emulate_action(
            instance_id,
            parent_branch_id="root",
            branch_id="branch-001",
            rng_id=1,
            decision_point_id="d-root-001",
            action_id="a-001",
            simulation_options={"stop_condition": "next_decision"},
            timeout_s=1.0,
        )
        self.assertEqual(response["branch_id"], "branch-001")
        self.assertEqual(self.requests[-1]["request_seq"], 2)

    async def test_rejected_response_consumes_sequence_and_allows_next_request(self) -> None:
        with self.assertRaises(RequestRejectedError):
            await self.client.start_instance(
                {"instance_type": "reject"}, timeout_s=1.0
            )
        self.assertEqual(self.client.next_request_seq, 2)
        self.assertIsNone(self.client.pending_retry)

        instance_id = await self.client.start_instance(
            {"instance_type": "combat"}, timeout_s=1.0
        )
        self.assertEqual(instance_id, "inst-001")
        self.assertEqual(self.requests[-1]["request_seq"], 2)

    async def test_envelope_mismatch_invalidates_stream_and_keeps_sequence_pending(self) -> None:
        with self.assertRaisesRegex(ApiProtocolError, "request_id"):
            await self.client.start_instance(
                {"instance_type": "mismatch"}, timeout_s=1.0
            )
        self.assertFalse(self.connection.is_alive())
        self.assertTrue(self.client.start_uncertain)
        self.assertEqual(self.client.pending_retry.request_seq, 1)
        self.assertEqual(self.client.next_request_seq, 1)

    async def test_operation_specific_validation_failure_does_not_advance_sequence(self) -> None:
        with self.assertRaisesRegex(ApiProtocolError, "instance_id"):
            await self.client.start_instance(
                {"instance_type": "missing-instance"}, timeout_s=1.0
            )
        self.assertFalse(self.connection.is_alive())
        self.assertTrue(self.client.start_uncertain)
        self.assertEqual(self.client.pending_retry.request_seq, 1)
        self.assertEqual(self.client.next_request_seq, 1)

    async def test_pending_protocol_failure_blocks_fresh_request(self) -> None:
        with self.assertRaises(ApiProtocolError):
            await self.client.start_instance(
                {"instance_type": "mismatch"}, timeout_s=1.0
            )
        with self.assertRaisesRegex(RuntimeError, "unresolved request"):
            await self.client.start_instance(
                {"instance_type": "combat"}, timeout_s=1.0
            )


if __name__ == "__main__":
    unittest.main()
