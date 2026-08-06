from __future__ import annotations

import pytest

from sts2_training.api.client import TrainingApiClient
from sts2_training.api.local_process_transport import LocalProcessTransport

pytestmark = pytest.mark.integration

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

WHOLE_RUN_CONFIG = {
    "instance_type": "whole_run",
    "character_id": "IRONCLAD",
    "ascension": 10,
    "seed": 18,
}


def _legal_actions(response: dict) -> list[dict]:
    actions = response["masked_emulator_dto"].get("legal_actions")
    assert isinstance(actions, list)
    assert actions
    return actions


def test_combat_root_branch_commit_and_close(
    api_client: TrainingApiClient,
    local_transport: LocalProcessTransport,
) -> None:
    assert local_transport.pid is not None
    assert local_transport.is_alive()

    # start_instance reaches CoreCLR, Sts2Emulator.dll, GameInstance, and the first
    # real Emulator decision. A separate emulated branch then exercises the worker path.
    instance_id = api_client.start_instance(COMBAT_CONFIG, timeout_s=120.0)
    before = api_client.get_decision(instance_id, "root", timeout_s=120.0)
    assert before["status"] == "completed"
    action_id = _legal_actions(before)[0]["action_id"]

    branch_id = "branch-smoke-001"
    branch = api_client.emulate_action(
        instance_id,
        parent_branch_id="root",
        branch_id=branch_id,
        rng_id=1,
        decision_point_id=before["decision_point_id"],
        action_id=action_id,
        simulation_options={
            "max_time_ms": 120_000,
            "stop_condition": "next_decision",
        },
        timeout_s=150.0,
    )
    assert branch["status"] == "completed"
    assert branch["branch_id"] == branch_id

    status = api_client.get_branch_status(
        instance_id,
        [branch_id],
        timeout_s=120.0,
    )
    assert status["branch_statuses"][branch_id] == "completed"

    committed = api_client.commit_action(
        instance_id,
        before["decision_point_id"],
        action_id,
        timeout_s=120.0,
    )
    assert committed["status"] == "completed"

    # Committing the root decision must release every speculative branch derived
    # from that decision. This verifies cleanup without a second explicit release.
    status = api_client.get_branch_status(
        instance_id,
        [branch_id],
        timeout_s=120.0,
    )
    assert status["branch_statuses"][branch_id] == "released"

    after = api_client.get_decision(instance_id, "root", timeout_s=120.0)
    assert after["status"] == "completed"
    assert after["decision_point_id"] != before["decision_point_id"]

    closed = api_client.close_instance(instance_id, timeout_s=120.0)
    assert closed["status"] == "completed"

    local_transport.close()
    assert not local_transport.is_alive()


def test_whole_run_start_decide_and_close(
    api_client: TrainingApiClient,
) -> None:
    instance_id = api_client.start_instance(WHOLE_RUN_CONFIG, timeout_s=120.0)
    decision = api_client.get_decision(instance_id, "root", timeout_s=120.0)
    assert decision["status"] == "completed"
    _legal_actions(decision)
    closed = api_client.close_instance(instance_id, timeout_s=120.0)
    assert closed["status"] == "completed"
