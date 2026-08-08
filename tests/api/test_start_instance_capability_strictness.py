from __future__ import annotations

from types import SimpleNamespace

import pytest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.contract import ApiProtocolError


def _client() -> AsyncTrainingApiClient:
    connection = SimpleNamespace(client_session_id="session-a")
    return AsyncTrainingApiClient(connection)  # type: ignore[arg-type]


def test_combat_start_requires_max_emulate_actions_items() -> None:
    client = _client()
    request = {"instance_config": {"instance_type": "combat"}}
    response = {"status": "completed", "instance_id": "inst-001"}

    with pytest.raises(ApiProtocolError, match="max_emulate_actions_items"):
        client._accept_start_instance_for_request(request, response)  # noqa: SLF001

    assert client.instance_id is None
    assert client.max_emulate_actions_items is None


def test_combat_start_accepts_positive_batch_capacity() -> None:
    client = _client()
    request = {"instance_config": {"instance_type": "combat"}}
    response = {
        "status": "completed",
        "instance_id": "inst-001",
        "max_emulate_actions_items": 16,
    }

    assert client._accept_start_instance_for_request(request, response) == "inst-001"  # noqa: SLF001
    assert client.max_emulate_actions_items == 16


def test_non_combat_start_may_omit_combat_batch_capacity() -> None:
    client = _client()
    request = {"instance_config": {"instance_type": "run"}}
    response = {"status": "completed", "instance_id": "inst-001"}

    assert client._accept_start_instance_for_request(request, response) == "inst-001"  # noqa: SLF001
    assert client.max_emulate_actions_items is None
