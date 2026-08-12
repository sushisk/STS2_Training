from __future__ import annotations

import json
from pathlib import Path

import pytest

from sts2_training.board_eval.card_features import CardFeatureExtractor
from sts2_training.board_eval.training_data import (
    MODEL_FEATURE_NAMES,
    NON_DECK_FEATURE_NAMES,
    build_examples_from_log,
    iter_log_events,
    label_from_events,
)


def _extractor() -> CardFeatureExtractor:
    return CardFeatureExtractor(
        [
            {
                "card_id": "STRIKE",
                "card_type": "Attack",
                "target_type": "AnyEnemy",
                "energy_cost": "1",
                "damage": "6",
            },
            {
                "card_id": "DEFEND",
                "card_type": "Skill",
                "target_type": "Self",
                "energy_cost": "1",
                "block": "5",
            },
        ]
    )


def _decision_event(
    decision_point_id: str,
    deck_ids: list[str],
    *,
    hp: int = 50,
    gold: int | None = 99,
    state_kind: str | None = "Shop",
    boundary: str | None = None,
) -> dict:
    dto: dict[str, object] = {
        "hp": hp,
        "maxHp": 80,
        "actFloor": 5,
        "totalFloor": 51,
        "deck": [{"id": card_id} for card_id in deck_ids],
    }
    if gold is not None:
        dto["gold"] = gold
    if state_kind is not None:
        dto["currentRoomType"] = state_kind
    event = {
        "event": "selection",
        "received": {"masked_emulator_dto": dto},
        "request": {"decision_point_id": decision_point_id, "operation": "commit_action"},
        "result": {"status": "completed"},
    }
    if boundary is not None:
        event["boundary"] = boundary
    return event


def _terminal_event(outcome: str) -> dict:
    return {
        "event": "selection",
        "received": None,
        "request": {"decision_point_id": "d-last", "operation": "commit_action"},
        "result": {"status": "completed", "masked_emulator_dto": {"outcome": outcome}},
        "run_result": {"outcome": outcome},
    }


def _write_jsonl(path: Path, events: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


def test_iter_log_events_and_label_from_events(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    _write_jsonl(path, [_decision_event("d0", ["STRIKE"]), _terminal_event("victory")])

    events = list(iter_log_events(path))

    assert label_from_events(events) == 1


def test_label_from_events_reads_current_self_play_terminal_record() -> None:
    events = [
        _decision_event("d0", ["STRIKE"]),
        {
            "event": "self_play_run_result",
            "run_id": "run-1",
            "outcome": "defeat",
            "final_dto": {"outcome": "defeat"},
        },
    ]

    assert label_from_events(events) == 0


def test_iter_log_events_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        list(iter_log_events(path))


def test_build_examples_keeps_run_label_and_state_kind(tmp_path: Path) -> None:
    path = tmp_path / "run-001.jsonl"
    _write_jsonl(
        path,
        [
            _decision_event("d0", ["STRIKE", "DEFEND"], state_kind="Shop"),
            _decision_event("d1", ["STRIKE"], hp=35, state_kind="RestSite"),
            _terminal_event("defeat"),
        ],
    )

    examples = build_examples_from_log(path, _extractor())

    assert [example.state_kind for example in examples] == ["Shop", "RestSite"]
    assert all(example.label == 0 for example in examples)
    assert all(example.run_id == "run-001" for example in examples)


def test_event_boundary_falls_back_when_current_room_type_is_missing(tmp_path: Path) -> None:
    path = tmp_path / "run-boundary.jsonl"
    _write_jsonl(
        path,
        [
            _decision_event(
                "d0",
                ["STRIKE"],
                state_kind=None,
                boundary="map_select",
            ),
            _terminal_event("victory"),
        ],
    )

    examples = build_examples_from_log(path, _extractor())

    assert [example.state_kind for example in examples] == ["map_select"]


def test_state_kind_filter_is_optional_foundation(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    _write_jsonl(
        path,
        [
            _decision_event("d0", ["STRIKE"], state_kind="Shop"),
            _decision_event("d1", ["STRIKE"], state_kind="RestSite"),
            _terminal_event("victory"),
        ],
    )

    unfiltered = build_examples_from_log(path, _extractor())
    shops_only = build_examples_from_log(path, _extractor(), state_kinds={"Shop"})

    assert len(unfiltered) == 2
    assert [example.state_kind for example in shops_only] == ["Shop"]


def test_state_kind_filter_accepts_event_boundary_fallback(tmp_path: Path) -> None:
    path = tmp_path / "run-boundary-filter.jsonl"
    _write_jsonl(
        path,
        [
            _decision_event("d0", ["STRIKE"], state_kind=None, boundary="map_select"),
            _decision_event("d1", ["STRIKE"], state_kind=None, boundary="reward_select"),
            _terminal_event("victory"),
        ],
    )

    map_only = build_examples_from_log(path, _extractor(), state_kinds={"map_select"})

    assert [example.state_kind for example in map_only] == ["map_select"]


def test_unknown_cards_remain_visible_in_skip_mode(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    _write_jsonl(
        path,
        [_decision_event("d0", ["STRIKE", "UNKNOWN"]), _terminal_event("victory")],
    )

    example = build_examples_from_log(path, _extractor(), on_unknown_card="skip")[0]

    assert example.deck_summary.deck_size == 2
    assert example.deck_summary.unknown_card_count == 1
    assert example.deck_summary.known_card_ratio == pytest.approx(0.5)


def test_missing_value_flag_distinguishes_missing_from_real_zero(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.jsonl"
    zero_path = tmp_path / "zero.jsonl"
    _write_jsonl(
        missing_path,
        [_decision_event("d0", ["STRIKE"], gold=None), _terminal_event("victory")],
    )
    _write_jsonl(
        zero_path,
        [_decision_event("d0", ["STRIKE"], gold=0), _terminal_event("victory")],
    )

    missing = build_examples_from_log(missing_path, _extractor())[0].to_model_vector()
    zero = build_examples_from_log(zero_path, _extractor())[0].to_model_vector()

    gold_index = MODEL_FEATURE_NAMES.index("gold")
    missing_index = MODEL_FEATURE_NAMES.index("gold_missing")
    assert missing[gold_index] == zero[gold_index] == 0.0
    assert missing[missing_index] == 1.0
    assert zero[missing_index] == 0.0


def test_model_feature_order_pairs_values_and_missing_flags(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    _write_jsonl(path, [_decision_event("d0", ["STRIKE"]), _terminal_event("victory")])

    example = build_examples_from_log(path, _extractor())[0]
    vector = example.to_model_vector()

    assert NON_DECK_FEATURE_NAMES == (
        "hp",
        "hp_missing",
        "max_hp",
        "max_hp_missing",
        "gold",
        "gold_missing",
        "act_floor",
        "act_floor_missing",
        "total_floor",
        "total_floor_missing",
    )
    assert len(vector) == len(MODEL_FEATURE_NAMES)
    assert vector[MODEL_FEATURE_NAMES.index("deck_deck_size")] == 1.0


def test_unlabeled_run_returns_no_examples(tmp_path: Path) -> None:
    path = tmp_path / "crashed.jsonl"
    _write_jsonl(path, [_decision_event("d0", ["STRIKE"])])

    assert build_examples_from_log(path, _extractor()) == []
