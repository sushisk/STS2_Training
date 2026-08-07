"""Compatibility adapter for the original asyncio transport API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sts2_training.api.tcp_connection import TcpConnection
from sts2_training.api.transport import JsonObject


class AsyncioTcpTransport(TcpConnection):
    """Backward-compatible name for callers from the minimum TCP transport step.

    New API code should depend on ``TcpConnection.exchange()`` directly. ``call()`` is
    retained only so existing smoke/tests and external callers are not broken.
    """

    async def call(
        self,
        request: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> JsonObject:
        return await self.exchange(request, timeout_s=timeout_s)
