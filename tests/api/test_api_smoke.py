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

RNG_COMBAT_CONFIG = {
    **COMBAT_CONFIG,
    "player_hp": 120,
    "player_max_hp": 120,
    "hand": ["DEFEND_IRONCLAD"],
    "draw_pile": (
        ["STRIKE_IRONCLAD"] * 4
        + ["DEFEND_IRONCLAD"] * 4
        + ["BASH"] * 2
    ),
    "seed": 7,
    "enemies": [{"monster_id": "CALCIFIED_CULTIST", "hp": 160}],
}

WHOLE_RUN_CONFIG = {
    "instance_type": "whole_run",
    "character_id": "IRONCLAD",
    "ascension": 10,
    "seed": 18,
}


def _legal_actions(response: dict) -> list[dict]:
    actions = response["masked_emulator_dto"].get("legal_actions")
    assert isinstance(actions, list) and actions, response
    return actions


def _action(
    response: dict,
    *,
    action_type: str | None = None,
    card_id: str | None = None,
) -> dict:
    for action in _legal_actions(response):
        parameters = action.get("parameters") or {}
        if action_type not in (None, action.get("action_type")):
            continue
        if card_id not in (None, parameters.get("cardId")):
            continue
        return action
    raise AssertionError(
        f"no action matched action_type={action_type!r}, card_id={card_id!r}"
    )


def _emulate(
    api_client: TrainingApiClient,
    instance_id: str,
    *,
    parent: str,
    branch: str,
    rng_id: int,
    decision: dict,
    action: dict,
) -> dict:
    response = api_client.emulate_action(
        instance_id,
        parent_branch_id=parent,
        branch_id=branch,
        rng_id=rng_id,
        decision_point_id=decision["decision_point_id"],
        action_id=action["action_id"],
        simulation_options={
            "max_time_ms": 120_000,
            "stop_condition": "next_decision",
        },
        timeout_s=150.0,
    )
    assert response["status"] == "completed", response
    return response


def _hand_card_ids(response: dict) -> list[str]:
    hand = response["masked_emulator_dto"].get("hand")
    assert isinstance(hand, list), response
    card_ids = [
        card.get("id") or card.get("cardId")
        for card in hand
        if isinstance(card, dict)
    ]
    assert len(card_ids) == len(hand)
    assert all(isinstance(card_id, str) for card_id in card_ids)
    return card_ids


def _advance_to_map(api_client: TrainingApiClient, instance_id: str) -> dict:
    decision = api_client.get_decision(instance_id, "root", timeout_s=120.0)
    for _ in range(50):
        if decision["masked_emulator_dto"].get("boundary") == "map_select":
            return decision
        decision = api_client.commit_action(
            instance_id,
            decision["decision_point_id"],
            _legal_actions(decision)[0]["action_id"],
            timeout_s=120.0,
        )
        assert decision["status"] == "completed", decision
    raise AssertionError("whole_run did not reach map_select within 50 decisions")


def test_combat_nested_branch_rng_lineage_commit_and_close(
    api_client: TrainingApiClient,
    local_transport: LocalProcessTransport,
) -> None:
    assert local_transport.pid is not None
    assert local_transport.is_alive()

    instance_id = api_client.start_instance(COMBAT_CONFIG, timeout_s=120.0)
    before = api_client.get_decision(instance_id, "root", timeout_s=120.0)
    root_action = _action(before, card_id="DEFEND_IRONCLAD")

    parent = _emulate(
        api_client,
        instance_id,
        parent="root",
        branch="branch-parent",
        rng_id=1,
        decision=before,
        action=root_action,
    )
    child_action = _action(parent, action_type="card")
    child = _emulate(
        api_client,
        instance_id,
        parent="branch-parent",
        branch="branch-child",
        rng_id=1,
        decision=parent,
        action=child_action,
    )
    assert child["parent_branch_id"] == "branch-parent"
    assert [entry["rng_id"] for entry in child["branch_log"]] == [1, 1]

    with pytest.raises(RequestRejectedError, match="requires rng_id=1"):
        _emulate(
            api_client,
            instance_id,
            parent="branch-parent",
            branch="branch-wrong-rng",
            rng_id=2,
            decision=parent,
            action=child_action,
        )

    committed = api_client.commit_action(
        instance_id,
        before["decision_point_id"],
        root_action["action_id"],
        timeout_s=120.0,
    )
    assert committed["status"] == "completed"

    statuses = api_client.get_branch_status(
        instance_id,
        ["branch-parent", "branch-child"],
        timeout_s=120.0,
    )
    assert set(statuses["branch_statuses"].values()) == {"released"}

    assert api_client.close_instance(instance_id, timeout_s=120.0)["status"] == "completed"
    local_transport.close()
    assert not local_transport.is_alive()


def test_combat_rng_ids_are_reproducible_and_change_draws(
    api_client: TrainingApiClient,
) -> None:
    instance_id = api_client.start_instance(RNG_COMBAT_CONFIG, timeout_s=120.0)
    try:
        root = api_client.get_decision(instance_id, "root", timeout_s=120.0)
        end_turn = _action(root, action_type="system")
        branches = [
            _emulate(
                api_client,
                instance_id,
                parent="root",
                branch=branch,
                rng_id=rng_id,
                decision=root,
                action=end_turn,
            )
            for branch, rng_id in (("rng-1a", 1), ("rng-1b", 1), ("rng-2", 2))
        ]

        first, repeated, changed = map(_hand_card_ids, branches)
        assert first == repeated
        assert first != changed
    finally:
        if api_client.instance_id == instance_id:
            api_client.close_instance(instance_id, timeout_s=120.0)


def test_whole_run_map_rejects_branch_and_close(
    api_client: TrainingApiClient,
) -> None:
    instance_id = api_client.start_instance(WHOLE_RUN_CONFIG, timeout_s=120.0)
    try:
        decision = _advance_to_map(api_client, instance_id)
        with pytest.raises(RequestRejectedError) as exc_info:
            api_client.emulate_action(
                instance_id,
                parent_branch_id="root",
                branch_id="whole-run-map-branch",
                rng_id=1,
                decision_point_id=decision["decision_point_id"],
                action_id=_legal_actions(decision)[0]["action_id"],
                simulation_options={"stop_condition": "next_decision"},
                timeout_s=120.0,
            )
        assert (
            exc_info.value.response["fault_kind"]
            == "rng_hypothesis_unsupported_at_boundary"
        )
    finally:
        if api_client.instance_id == instance_id:
            assert (
                api_client.close_instance(instance_id, timeout_s=120.0)["status"]
                == "completed"
            )
