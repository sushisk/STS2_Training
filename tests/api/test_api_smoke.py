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


def _action_id(response: dict, card_id: str | None = None) -> str:
    actions = _legal_actions(response)
    cards = [
        action
        for action in actions
        if action.get("action_type") == "card"
        and (
            card_id is None
            or (action.get("parameters") or {}).get("cardId") == card_id
        )
    ]
    if card_id is not None:
        assert len(cards) == 1, cards
    return (cards[0] if cards else actions[0])["action_id"]


def test_combat_nested_branches_rng_lineage_commit_and_close(
    api_client: TrainingApiClient,
    local_transport: LocalProcessTransport,
) -> None:
    assert local_transport.pid is not None
    assert local_transport.is_alive()

    instance_id = api_client.start_instance(COMBAT_CONFIG, timeout_s=120.0)
    root = api_client.get_decision(instance_id, "root", timeout_s=120.0)
    assert root["status"] == "completed"
    root_action_id = _action_id(root, "DEFEND_IRONCLAD")

    branches = {}
    for rng_id in (1, 2):
        branch_id = f"branch-rng-{rng_id}"
        branches[branch_id] = api_client.emulate_action(
            instance_id,
            parent_branch_id="root",
            branch_id=branch_id,
            rng_id=rng_id,
            decision_point_id=root["decision_point_id"],
            action_id=root_action_id,
            simulation_options={
                "max_time_ms": 120_000,
                "stop_condition": "next_decision",
            },
            timeout_s=150.0,
        )
        assert branches[branch_id]["status"] == "completed"
        assert branches[branch_id]["rng_id"] == rng_id

    parent = branches["branch-rng-1"]
    child_action_id = _action_id(parent)
    child = api_client.emulate_action(
        instance_id,
        parent_branch_id="branch-rng-1",
        branch_id="branch-child",
        rng_id=1,
        decision_point_id=parent["decision_point_id"],
        action_id=child_action_id,
        simulation_options={
            "max_time_ms": 120_000,
            "stop_condition": "next_decision",
        },
        timeout_s=150.0,
    )
    assert child["status"] == "completed"
    assert child["parent_branch_id"] == "branch-rng-1"

    with pytest.raises(RequestRejectedError):
        api_client.emulate_action(
            instance_id,
            parent_branch_id="branch-rng-1",
            branch_id="branch-child-wrong-rng",
            rng_id=2,
            decision_point_id=parent["decision_point_id"],
            action_id=child_action_id,
            timeout_s=120.0,
        )

    branch_ids = [*branches, "branch-child"]
    status = api_client.get_branch_status(
        instance_id,
        branch_ids,
        timeout_s=120.0,
    )
    assert status["branch_statuses"] == {
        branch_id: "completed" for branch_id in branch_ids
    }

    committed = api_client.commit_action(
        instance_id,
        root["decision_point_id"],
        root_action_id,
        timeout_s=120.0,
    )
    assert committed["status"] == "completed"

    status = api_client.get_branch_status(
        instance_id,
        branch_ids,
        timeout_s=120.0,
    )
    assert status["branch_statuses"] == {
        branch_id: "released" for branch_id in branch_ids
    }

    after = api_client.get_decision(instance_id, "root", timeout_s=120.0)
    assert after["decision_point_id"] != root["decision_point_id"]

    closed = api_client.close_instance(instance_id, timeout_s=120.0)
    assert closed["status"] == "completed"

    local_transport.close()
    assert not local_transport.is_alive()


def test_whole_run_map_rejects_branch_and_closes(
    api_client: TrainingApiClient,
) -> None:
    instance_id = api_client.start_instance(WHOLE_RUN_CONFIG, timeout_s=120.0)
    decision = api_client.get_decision(instance_id, "root", timeout_s=120.0)

    for _ in range(50):
        if decision["masked_emulator_dto"].get("boundary") == "map_select":
            break
        decision = api_client.commit_action(
            instance_id,
            decision["decision_point_id"],
            _action_id(decision),
            timeout_s=120.0,
        )
    else:
        raise AssertionError("whole_run did not reach map_select")

    with pytest.raises(RequestRejectedError) as exc_info:
        api_client.emulate_action(
            instance_id,
            parent_branch_id="root",
            branch_id="whole-run-map-branch",
            rng_id=1,
            decision_point_id=decision["decision_point_id"],
            action_id=_legal_actions(decision)[0]["action_id"],
            timeout_s=120.0,
        )
    assert (
        exc_info.value.response["fault_kind"]
        == "rng_hypothesis_unsupported_at_boundary"
    )

    closed = api_client.close_instance(instance_id, timeout_s=120.0)
    assert closed["status"] == "completed"
