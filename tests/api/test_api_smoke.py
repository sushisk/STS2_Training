from __future__ import annotations

import json
import random

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

RANDOM_COMBAT_CONFIG = {
    "instance_type": "combat",
    "character_id": "IRONCLAD",
    "player_hp": 120,
    "player_max_hp": 120,
    "hand": [
        "STRIKE_IRONCLAD",
        "DEFEND_IRONCLAD",
        "BASH",
    ],
    "draw_pile": [
        "STRIKE_IRONCLAD",
        "DEFEND_IRONCLAD",
        "STRIKE_IRONCLAD",
        "DEFEND_IRONCLAD",
        "STRIKE_IRONCLAD",
        "DEFEND_IRONCLAD",
        "BASH",
        "STRIKE_IRONCLAD",
        "DEFEND_IRONCLAD",
    ],
    "discard_pile": [],
    "exhaust_pile": [],
    "player_powers": [],
    "relics": [],
    "potions": [],
    "seed": 7,
    "enemies": [{"monster_id": "CALCIFIED_CULTIST", "hp": 160}],
}

RANDOM_WHOLE_RUN_CONFIG = {
    "instance_type": "whole_run",
    "character_id": "IRONCLAD",
    "ascension": 0,
    "seed": 18,
}


def _legal_actions(response: dict) -> list[dict]:
    actions = response["masked_emulator_dto"].get("legal_actions")
    assert isinstance(actions, list)
    assert actions
    return actions


def _random_root_walk(
    api_client: TrainingApiClient,
    instance_config: dict,
    *,
    random_seed: int,
    max_decisions: int,
    min_decisions: int,
) -> list[dict]:
    """Advance only root by choosing directly from each published legal-action list."""
    rng = random.Random(random_seed)
    instance_id = api_client.start_instance(instance_config, timeout_s=120.0)
    decisions: list[dict] = []

    try:
        current = api_client.get_decision(instance_id, "root", timeout_s=120.0)
        assert current["status"] == "completed"

        for step_index in range(max_decisions):
            actions = current["masked_emulator_dto"].get("legal_actions")
            if not actions:
                break
            assert isinstance(actions, list)

            chosen = rng.choice(actions)
            previous_decision_point_id = current["decision_point_id"]
            current = api_client.commit_action(
                instance_id,
                previous_decision_point_id,
                chosen["action_id"],
                timeout_s=120.0,
            )

            assert current["status"] == "completed"
            assert current["branch_id"] == "root"
            assert current["decision_point_id"] != previous_decision_point_id

            branch_log = current["branch_log"]
            assert len(branch_log) == step_index + 1
            assert branch_log[-1] == {
                "depth": step_index,
                "decision_point_id": previous_decision_point_id,
                "action_id": chosen["action_id"],
                "rng_id": 0,
            }
            decisions.append(current)

        assert len(decisions) >= min_decisions
        assert len({decision["decision_point_id"] for decision in decisions}) == len(
            decisions
        )
        public_states = {
            json.dumps(
                decision["masked_emulator_dto"],
                sort_keys=True,
                separators=(",", ":"),
            )
            for decision in decisions
        }
        assert len(public_states) >= 2
        return decisions
    finally:
        if api_client.instance_id == instance_id:
            closed = api_client.close_instance(instance_id, timeout_s=120.0)
            assert closed["status"] == "completed"


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


def test_combat_random_root_actions_advance_public_state(
    api_client: TrainingApiClient,
) -> None:
    _random_root_walk(
        api_client,
        RANDOM_COMBAT_CONFIG,
        random_seed=20260807,
        max_decisions=12,
        min_decisions=8,
    )


def test_whole_run_random_root_actions_advance_public_state(
    api_client: TrainingApiClient,
) -> None:
    decisions = _random_root_walk(
        api_client,
        RANDOM_WHOLE_RUN_CONFIG,
        random_seed=20260808,
        max_decisions=16,
        min_decisions=10,
    )
    assert any(
        "boundary" in decision["masked_emulator_dto"] for decision in decisions
    )
