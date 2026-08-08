"""Regression coverage for the documented Training timeout scope."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any

from sts2_training.api.async_client import AsyncTrainingApiClient


class _ImmediateConnection:
    client_session_id = "timeout-scope-session"

    async def exchange(
        self,
        message: Mapping[str, Any],
        *,
        deadline: float,
    ) -> dict[str, Any]:
        # The complete response frame is conceptually available immediately, before the
        # absolute I/O deadline. Local DTO validation happens after exchange() returns.
        return {
            **dict(message),
            "server_epoch": "epoch",
            "status": "completed",
            "branch_statuses": {"branch-1": "released"},
        }


def test_timeout_s_excludes_synchronous_post_frame_validation() -> None:
    async def _scenario() -> None:
        client = AsyncTrainingApiClient(_ImmediateConnection())  # type: ignore[arg-type]
        client._instance_id = "instance"  # noqa: SLF001
        original_validate = client._validate_api_response  # noqa: SLF001

        def _slow_validate(request, response):
            time.sleep(0.03)
            return original_validate(request, response)

        client._validate_api_response = _slow_validate  # type: ignore[method-assign]  # noqa: SLF001
        started = time.perf_counter()
        result = await client.get_branch_status(
            "instance",
            ["branch-1"],
            timeout_s=0.005,
        )
        elapsed = time.perf_counter() - started

        assert result["branch_statuses"] == {"branch-1": "released"}
        assert elapsed >= 0.02
        assert client.next_request_seq == 2
        assert client.pending_retry is None

    asyncio.run(_scenario())
