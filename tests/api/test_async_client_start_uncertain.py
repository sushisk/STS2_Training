from __future__ import annotations

import asyncio
import unittest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.transport import TransportError


class FailingStartConnection:
    def __init__(self, *, completion_uncertain: bool) -> None:
        self.exchange_calls = 0
        self.completion_uncertain = completion_uncertain

    async def exchange(self, message, *, deadline: float):
        self.exchange_calls += 1
        raise TransportError(
            "simulated transport failure",
            completion_uncertain=self.completion_uncertain,
        )

    async def invalidate(self) -> None:
        pass

    async def close(self) -> None:
        pass


class CancelledStartConnection:
    def __init__(self, *, completion_uncertain: bool) -> None:
        self.exchange_calls = 0
        self.completion_uncertain = completion_uncertain

    async def exchange(self, message, *, deadline: float):
        self.exchange_calls += 1
        exc = asyncio.CancelledError()
        setattr(exc, "completion_uncertain", self.completion_uncertain)
        raise exc

    async def invalidate(self) -> None:
        pass

    async def close(self) -> None:
        pass


class AsyncTrainingApiClientStartUncertainTest(unittest.IsolatedAsyncioTestCase):
    async def test_uncertain_transport_failure_blocks_later_start(self) -> None:
        connection = FailingStartConnection(completion_uncertain=True)
        client = AsyncTrainingApiClient(connection)

        with self.assertRaisesRegex(TransportError, "simulated transport failure"):
            await client.start_instance(
                {"instance_type": "combat"},
                timeout_s=0.01,
            )

        self.assertTrue(client.start_uncertain)
        self.assertIsNone(client.instance_id)

        with self.assertRaisesRegex(RuntimeError, "result is unknown"):
            await client.start_instance(
                {"instance_type": "combat"},
                timeout_s=1.0,
            )

        self.assertEqual(connection.exchange_calls, 1)

    async def test_known_pre_send_transport_failure_does_not_mark_uncertain(self) -> None:
        connection = FailingStartConnection(completion_uncertain=False)
        client = AsyncTrainingApiClient(connection)

        for _ in range(2):
            with self.assertRaisesRegex(TransportError, "simulated transport failure"):
                await client.start_instance(
                    {"instance_type": "combat"},
                    timeout_s=1.0,
                )

        self.assertFalse(client.start_uncertain)
        self.assertIsNone(client.instance_id)
        self.assertEqual(connection.exchange_calls, 2)

    async def test_uncertain_cancellation_blocks_later_start(self) -> None:
        connection = CancelledStartConnection(completion_uncertain=True)
        client = AsyncTrainingApiClient(connection)

        with self.assertRaises(asyncio.CancelledError):
            await client.start_instance(
                {"instance_type": "combat"},
                timeout_s=1.0,
            )

        self.assertTrue(client.start_uncertain)

        with self.assertRaisesRegex(RuntimeError, "result is unknown"):
            await client.start_instance(
                {"instance_type": "combat"},
                timeout_s=1.0,
            )

        self.assertEqual(connection.exchange_calls, 1)

    async def test_known_pre_send_cancellation_does_not_mark_uncertain(self) -> None:
        connection = CancelledStartConnection(completion_uncertain=False)
        client = AsyncTrainingApiClient(connection)

        for _ in range(2):
            with self.assertRaises(asyncio.CancelledError):
                await client.start_instance(
                    {"instance_type": "combat"},
                    timeout_s=1.0,
                )

        self.assertFalse(client.start_uncertain)
        self.assertEqual(connection.exchange_calls, 2)


if __name__ == "__main__":
    unittest.main()
