from __future__ import annotations

import asyncio
import unittest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.transport import TransportError


class FailingStartConnection:
    def __init__(self) -> None:
        self.exchange_calls = 0

    async def exchange(self, message, *, timeout_s: float):
        self.exchange_calls += 1
        raise TransportError("simulated timeout")

    async def invalidate(self) -> None:
        pass

    async def close(self) -> None:
        pass


class CancelledStartConnection:
    def __init__(self) -> None:
        self.exchange_calls = 0

    async def exchange(self, message, *, timeout_s: float):
        self.exchange_calls += 1
        raise asyncio.CancelledError

    async def invalidate(self) -> None:
        pass

    async def close(self) -> None:
        pass


class AsyncTrainingApiClientStartUncertainTest(unittest.IsolatedAsyncioTestCase):
    async def test_transport_failure_blocks_later_start_without_sending(self) -> None:
        connection = FailingStartConnection()
        client = AsyncTrainingApiClient(connection)

        with self.assertRaisesRegex(TransportError, "simulated timeout"):
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

    async def test_cancellation_blocks_later_start_without_sending(self) -> None:
        connection = CancelledStartConnection()
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


if __name__ == "__main__":
    unittest.main()
