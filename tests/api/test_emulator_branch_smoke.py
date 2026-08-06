from __future__ import annotations

import pytest

from sts2_training.api.client import TrainingApiClient

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


def test_branch_worker_executes_one_emulator_step(
    api_client: TrainingApiClient,
) -> None:
    """Exercises the separate BranchWorkerPool process and its Emulator instance."""

    instance_id = api_client.start_instance(COMBAT_CONFIG, timeout_s=120.0)
    root = api_client.get_decision(instance_id, "root", timeout_s=120.0)
    legal_actions = root["masked_emulator_dto"].get("legal_actions")
    assert isinstance(legal_actions, list) and legal_actions

    branch_id = "branch-emulator-smoke"
    result = api_client.emulate_action(
        instance_id,
        parent_branch_id="root",
        branch_id=branch_id,
        rng_id=1,
        decision_point_id=root["decision_point_id"],
        action_id=legal_actions[0]["action_id"],
        simulation_options={
            "max_time_ms": 120_000,
            "stop_condition": "next_decision",
        },
        timeout_s=150.0,
    )

    assert result["status"] == "completed", result
    assert result["branch_id"] == branch_id
    assert result["parent_branch_id"] == "root"
    assert result["rng_id"] == 1
    assert isinstance(result["masked_emulator_dto"], dict)

    status = api_client.get_branch_status(
        instance_id,
        [branch_id],
        timeout_s=120.0,
    )
    assert status["status"] == "completed"
    assert status["branch_statuses"][branch_id] == "completed"

    released = api_client.release_branches(
        instance_id,
        [branch_id],
        timeout_s=120.0,
    )
    assert released["status"] == "completed"
    assert released["branch_statuses"][branch_id] == "released"

    closed = api_client.close_instance(instance_id, timeout_s=120.0)
    assert closed["status"] == "completed"
