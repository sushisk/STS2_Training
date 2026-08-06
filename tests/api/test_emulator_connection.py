from __future__ import annotations

import pytest

from sts2_training.api.local_process_transport import LocalProcessTransport

pytestmark = [pytest.mark.integration, pytest.mark.emulator]

COMBAT_CONFIG = {
    "instance_type": "combat",
    "character_id": "IRONCLAD",
    "player_hp": None,
    "player_max_hp": None,
    "hand": ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"],
    "draw_pile": [],
    "discard_pile": [],
    "exhaust_pile": [],
    "player_powers": [],
    "relics": [],
    "potions": [],
    "seed": 1,
    "enemies": [{"monster_id": "CALCIFIED_CULTIST", "hp": 48}],
}


def test_spawned_runtime_connects_to_emulator_and_returns_root_decision(
    local_transport: LocalProcessTransport,
) -> None:
    """Training -> RL child -> CoreCLR -> Emulator -> first legal decision."""

    assert local_transport.pid is not None
    assert local_transport.is_alive()

    start_response = local_transport.call(
        {
            "schema_version": "0.5",
            "request_id": "req-emulator-connect-001",
            "operation": "start_instance",
            "instance_config": COMBAT_CONFIG,
        },
        timeout_s=120.0,
    )

    assert start_response.get("status") == "completed", (
        "Failed before the Emulator produced its first decision. "
        f"RL response: {start_response!r}"
    )
    instance_id = start_response.get("instance_id")
    assert isinstance(instance_id, str) and instance_id
    assert start_response.get("branch_id") == "root"
    assert isinstance(start_response.get("decision_point_id"), str)

    masked_dto = start_response.get("masked_emulator_dto")
    assert isinstance(masked_dto, dict)
    legal_actions = masked_dto.get("legal_actions")
    assert isinstance(legal_actions, list)
    assert legal_actions, "Emulator connected, but returned no legal root actions"
    assert all(
        isinstance(action, dict) and isinstance(action.get("action_id"), str)
        for action in legal_actions
    )

    close_response = local_transport.call(
        {
            "schema_version": "0.5",
            "request_id": "req-emulator-connect-002",
            "operation": "close_instance",
            "instance_id": instance_id,
        },
        timeout_s=120.0,
    )
    assert close_response.get("status") == "completed", close_response
    assert local_transport.is_alive()
