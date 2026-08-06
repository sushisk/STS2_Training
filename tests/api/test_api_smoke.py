from __future__ import annotations

import pytest

from sts2_training.api.client import RequestRejectedError, TrainingApiClient
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


def _card_action_id(response: dict, card_id: str) -> str:
    return next(
        action["action_id"]
        for action in _legal_actions(response)
        if action.get("parameters", {}).get("cardId") == card_id
    )


def _safe_action_id(response: dict) -> str:
    actions = _legal_actions(response)
    for action in actions:
        if action.get("action_type") == "card":
            return action["action_id"]
    return actions[0]["action_id"]


def test_combat_real_process_branch_lifecycle_and_stale_decision_rejection(
    api_client: TrainingApiClient,
    local_transport: LocalProcessTransport,
) -> None:
    instance_id = api_client.start_instance(COMBAT_CONFIG, timeout_s=120.0)
    assert instance_id

    before = api_client.get_decision(instance_id, "root", timeout_s=120.0)
    assert before["status"] == "completed"
    defend_id = _card_action_id(before, "DEFEND_IRONCLAD")
    bash_id = _card_action_id(before, "BASH")
    root_action_ids = [action["action_id"] for action in _legal_actions(before)]

    branch_one = api_client.emulate_action(
        instance_id,
        "root",
        "branch-combat-001",
        1,
        before["decision_point_id"],
        defend_id,
        timeout_s=120.0,
    )
    branch_two = api_client.emulate_action(
        instance_id,
        "root",
        "branch-combat-002",
        1,
        before["decision_point_id"],
        bash_id,
        timeout_s=120.0,
    )
    assert branch_one["status"] == "completed"
    assert branch_two["status"] == "completed"

    deep_branch = api_client.emulate_action(
        instance_id,
        branch_one["branch_id"],
        "branch-combat-001-child-001",
        1,
        branch_one["decision_point_id"],
        _safe_action_id(branch_one),
        timeout_s=120.0,
    )
    assert deep_branch["status"] == "completed"

    root_after_emulation = api_client.get_decision(
        instance_id,
        "root",
        timeout_s=120.0,
    )
    assert root_after_emulation["decision_point_id"] == before["decision_point_id"]
    assert [
        action["action_id"] for action in _legal_actions(root_after_emulation)
    ] == root_action_ids

    discarded_branch_ids = [branch_two["branch_id"], deep_branch["branch_id"]]
    cancelled = api_client.cancel_branches(
        instance_id,
        discarded_branch_ids,
        timeout_s=120.0,
    )
    assert cancelled["status"] == "completed"
    released = api_client.release_branches(
        instance_id,
        discarded_branch_ids,
        timeout_s=120.0,
    )
    assert released["status"] == "completed"
    statuses = api_client.get_branch_status(
        instance_id,
        discarded_branch_ids,
        timeout_s=120.0,
    )["branch_statuses"]
    assert all(statuses[branch_id] == "released" for branch_id in discarded_branch_ids)

    committed = api_client.commit_action(
        instance_id,
        before["decision_point_id"],
        defend_id,
        timeout_s=120.0,
    )
    assert committed["status"] == "completed"
    assert committed["decision_point_id"] != before["decision_point_id"]

    with pytest.raises(RequestRejectedError):
        api_client.commit_action(
            instance_id,
            before["decision_point_id"],
            defend_id,
            timeout_s=120.0,
        )

    after_rejection = api_client.get_decision(instance_id, "root", timeout_s=120.0)
    assert after_rejection["decision_point_id"] == committed["decision_point_id"]
    branch_one_status = api_client.get_branch_status(
        instance_id,
        [branch_one["branch_id"]],
        timeout_s=120.0,
    )["branch_statuses"][branch_one["branch_id"]]
    assert branch_one_status == "released"

    closed = api_client.close_instance(instance_id, timeout_s=120.0)
    assert closed["status"] == "completed"

    local_transport.close()
    assert not local_transport.is_alive()


def test_whole_run_real_process_commit_advances_decision(
    api_client: TrainingApiClient,
) -> None:
    instance_id = api_client.start_instance(WHOLE_RUN_CONFIG, timeout_s=120.0)
    decision = api_client.get_decision(instance_id, "root", timeout_s=120.0)
    assert decision["status"] == "completed"

    committed = api_client.commit_action(
        instance_id,
        decision["decision_point_id"],
        _safe_action_id(decision),
        timeout_s=120.0,
    )
    assert committed["status"] == "completed"
    assert committed["decision_point_id"] != decision["decision_point_id"]

    next_decision = api_client.get_decision(instance_id, "root", timeout_s=120.0)
    assert next_decision["decision_point_id"] == committed["decision_point_id"]

    closed = api_client.close_instance(instance_id, timeout_s=120.0)
    assert closed["status"] == "completed"
