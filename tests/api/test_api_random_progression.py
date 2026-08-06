from __future__ import annotations

import json
import random
import sys
from typing import Any

import pytest

from sts2_training.api.client import TrainingApiClient

pytestmark = pytest.mark.integration

COMBAT_RANDOM_WALK_CONFIG = {
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

WHOLE_RUN_RANDOM_WALK_CONFIG = {
    "instance_type": "whole_run",
    "character_id": "IRONCLAD",
    "ascension": 0,
    "seed": 18,
}

_NON_BOARD_KEYS = frozenset({"dto_version", "mask_version", "legal_actions"})


def _masked_dto(response: dict[str, Any]) -> dict[str, Any]:
    dto = response.get("masked_emulator_dto")
    assert isinstance(dto, dict), f"missing masked_emulator_dto: {response!r}"
    return dto


def _available_actions(response: dict[str, Any]) -> list[dict[str, Any]]:
    actions = _masked_dto(response).get("legal_actions")
    assert isinstance(actions, list), f"legal_actions must be a list: {response!r}"
    for action in actions:
        assert isinstance(action, dict), f"legal action must be an object: {action!r}"
        assert isinstance(action.get("action_id"), str), (
            f"legal action must expose a string action_id: {action!r}"
        )
    return [action for action in actions if action.get("is_available") is not False]


def _board_fingerprint(response: dict[str, Any]) -> str:
    board = {
        key: value
        for key, value in _masked_dto(response).items()
        if key not in _NON_BOARD_KEYS
    }
    assert board, f"public board DTO is empty: {response!r}"
    return json.dumps(
        board,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _trace_json(trace: list[dict[str, Any]]) -> str:
    return json.dumps(trace, ensure_ascii=False, sort_keys=True, default=repr)


def _close_without_masking_primary_failure(
    api_client: TrainingApiClient,
    instance_id: str,
) -> None:
    primary_error = sys.exception()
    try:
        closed = api_client.close_instance(instance_id, timeout_s=120.0)
        assert closed["status"] == "completed", closed
    except BaseException as close_error:
        if primary_error is None:
            raise
        primary_error.add_note(f"close_instance also failed: {close_error!r}")


def _random_root_walk(
    api_client: TrainingApiClient,
    instance_config: dict[str, Any],
    *,
    random_seed: int,
    max_decisions: int,
    min_decisions: int,
    min_distinct_boards: int,
) -> dict[str, Any]:
    """Advance root only, selecting one published action without branch exploration."""
    rng = random.Random(random_seed)
    instance_id = api_client.start_instance(instance_config, timeout_s=120.0)
    trace: list[dict[str, Any]] = []

    try:
        initial = api_client.get_decision(instance_id, "root", timeout_s=120.0)
        assert initial["status"] == "completed", initial
        current = initial
        board_fingerprints = [_board_fingerprint(initial)]
        initial_decision_point_id = initial["decision_point_id"]
        assert isinstance(initial_decision_point_id, str) and initial_decision_point_id
        decision_point_ids = [initial_decision_point_id]

        for step_index in range(max_decisions):
            actions = _available_actions(current)
            if not actions:
                break

            chosen_offset = rng.randrange(len(actions))
            chosen = actions[chosen_offset]
            previous_decision_point_id = current["decision_point_id"]
            current = api_client.commit_action(
                instance_id,
                previous_decision_point_id,
                chosen["action_id"],
                timeout_s=120.0,
            )

            post_board_fingerprint = _board_fingerprint(current)
            trace_entry = {
                "step": step_index,
                "decision_point_id": previous_decision_point_id,
                "option_count": len(actions),
                "chosen_offset": chosen_offset,
                "action_id": chosen["action_id"],
                "action_type": chosen.get("action_type"),
                "label": chosen.get("label"),
                "post_boundary": _masked_dto(current).get("boundary"),
                "board_changed": post_board_fingerprint != board_fingerprints[-1],
            }
            trace.append(trace_entry)
            diagnostics = _trace_json(trace)

            assert current["status"] == "completed", diagnostics
            assert current["branch_id"] == "root", diagnostics
            assert current["decision_point_id"] != previous_decision_point_id, diagnostics

            branch_log = current["branch_log"]
            assert len(branch_log) == step_index + 1, diagnostics
            assert branch_log[-1] == {
                "depth": step_index,
                "decision_point_id": previous_decision_point_id,
                "action_id": chosen["action_id"],
                "rng_id": 0,
            }, diagnostics

            decision_point_ids.append(current["decision_point_id"])
            board_fingerprints.append(post_board_fingerprint)

        diagnostics = _trace_json(trace)
        assert len(trace) >= min_decisions, diagnostics
        assert len(set(decision_point_ids)) == len(decision_point_ids), diagnostics
        assert any(entry["option_count"] > 1 for entry in trace), diagnostics
        assert any(
            entry["option_count"] > 1 and entry["chosen_offset"] > 0
            for entry in trace
        ), diagnostics
        assert len(set(board_fingerprints)) >= min_distinct_boards, diagnostics

        return {
            "initial": initial,
            "final": current,
            "trace": trace,
            "board_fingerprints": board_fingerprints,
        }
    finally:
        if api_client.instance_id == instance_id:
            _close_without_masking_primary_failure(api_client, instance_id)


def test_combat_random_root_actions_advance_board(
    api_client: TrainingApiClient,
) -> None:
    result = _random_root_walk(
        api_client,
        COMBAT_RANDOM_WALK_CONFIG,
        random_seed=20260807,
        max_decisions=12,
        min_decisions=8,
        min_distinct_boards=4,
    )

    initial_board = _masked_dto(result["initial"])
    assert initial_board.get("hp") == 120
    enemies = initial_board.get("enemies")
    assert isinstance(enemies, list) and enemies
    assert enemies[0].get("hp") == 160


def test_whole_run_random_root_actions_advance_board(
    api_client: TrainingApiClient,
) -> None:
    result = _random_root_walk(
        api_client,
        WHOLE_RUN_RANDOM_WALK_CONFIG,
        random_seed=20260808,
        max_decisions=16,
        min_decisions=10,
        min_distinct_boards=4,
    )

    initial_board = _masked_dto(result["initial"])
    assert isinstance(initial_board.get("boundary"), str)
    assert isinstance(initial_board.get("room_context"), dict)
