from __future__ import annotations

import asyncio
import itertools
import json
import unittest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.asyncio_tcp_transport import AsyncioTcpTransport
from sts2_training.api.client import RequestRejectedError


def ids():
    counter = itertools.count(1)
    return lambda: f"req-{next(counter):03d}"


class AsyncTrainingApiClientTcpTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.requests: list[dict] = []
        self.server = await asyncio.start_server(
            self._handle_client,
            "127.0.0.1",
            0,
        )
        self.port = int(self.server.sockets[0].getsockname()[1])
        self.transport = AsyncioTcpTransport(port=self.port)
        self.client = AsyncTrainingApiClient(
            self.transport,
            request_id_factory=ids(),
        )

    async def asyncTearDown(self) -> None:
        await self.client.close()
        self.server.close()
        await self.server.wait_closed()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            while line := await reader.readline():
                request = json.loads(line)
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
    def _response_for(request: dict) -> dict:
        common = {
            "schema_version": request["schema_version"],
            "request_id": request["request_id"],
            "operation": request["operation"],
        }
        operation = request["operation"]

        if operation == "start_instance":
            if request["instance_config"].get("instance_type") == "reject":
                return {
                    **common,
                    "status": "rejected",
                    "error": "bad config",
                }
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

    async def test_start_instance_sends_v05_dto(self) -> None:
        instance_id = await self.client.start_instance(
            {"instance_type": "combat"},
            timeout_s=1.0,
        )

        self.assertEqual(instance_id, "inst-001")
        self.assertEqual(
            self.requests,
            [
                {
                    "schema_version": "0.5",
                    "request_id": "req-001",
                    "operation": "start_instance",
                    "instance_config": {"instance_type": "combat"},
                }
            ],
        )

    async def test_decision_and_commit_round_trip_over_tcp(self) -> None:
        instance_id = await self.client.start_instance(
            {"instance_type": "combat"},
            timeout_s=1.0,
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
        self.assertEqual(
            self.requests[-1],
            {
                "schema_version": "0.5",
                "request_id": "req-003",
                "operation": "commit_action",
                "instance_id": "inst-001",
                "branch_id": "root",
                "rng_id": 0,
                "decision_point_id": "d-root-001",
                "action_id": "a-001",
            },
        )

    async def test_emulate_action_round_trip_over_tcp(self) -> None:
        instance_id = await self.client.start_instance(
            {"instance_type": "combat"},
            timeout_s=1.0,
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
        self.assertEqual(response["rng_id"], 1)
        self.assertEqual(
            self.requests[-1]["simulation_options"],
            {"stop_condition": "next_decision"},
        )

    async def test_rejected_dto_response_raises_api_error(self) -> None:
        with self.assertRaises(RequestRejectedError):
            await self.client.start_instance(
                {"instance_type": "reject"},
                timeout_s=1.0,
            )


if __name__ == "__main__":
    unittest.main()
