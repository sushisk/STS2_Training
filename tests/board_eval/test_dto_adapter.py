from __future__ import annotations

import pytest

from sts2_training.board_eval.card_features import CardFeatureExtractor
from sts2_training.board_eval.dto_adapter import (
    DeckCardRef,
    board_context_from_dto,
    deck_card_refs_from_dto,
    deck_features_from_dto,
    deck_features_with_unknown_count_from_dto,
    deck_summary_from_dto,
    state_kind_from_dto,
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


def test_deck_card_refs_preserve_upgrade_and_duplicates() -> None:
    dto = {
        "deck": [
            {"id": "STRIKE"},
            {"id": "STRIKE", "upgraded": True},
            {"id": "DEFEND"},
        ]
    }

    assert deck_card_refs_from_dto(dto) == [
        DeckCardRef("STRIKE", False),
        DeckCardRef("STRIKE", True),
        DeckCardRef("DEFEND", False),
    ]


def test_deck_card_refs_validate_wire_shape() -> None:
    with pytest.raises(ValueError, match="sequence"):
        deck_card_refs_from_dto({"deck": "bad"})
    with pytest.raises(ValueError, match="mapping"):
        deck_card_refs_from_dto({"deck": ["STRIKE"]})
    with pytest.raises(ValueError, match="id"):
        deck_card_refs_from_dto({"deck": [{}]})
    with pytest.raises(ValueError, match="upgraded"):
        deck_card_refs_from_dto({"deck": [{"id": "STRIKE", "upgraded": "yes"}]})


def test_unknown_count_is_reported_when_skip_policy_is_used() -> None:
    cards, unknown_count = deck_features_with_unknown_count_from_dto(
        {"deck": [{"id": "STRIKE"}, {"id": "UNKNOWN"}, {"id": "UNKNOWN"}]},
        _extractor(),
        on_unknown_card="skip",
    )

    assert [card.card_id for card in cards] == ["STRIKE"]
    assert unknown_count == 2


def test_legacy_cards_only_helper_still_returns_cards() -> None:
    cards = deck_features_from_dto(
        {"deck": [{"id": "STRIKE"}, {"id": "UNKNOWN"}]},
        _extractor(),
        on_unknown_card="skip",
    )

    assert [card.card_id for card in cards] == ["STRIKE"]


def test_summary_preserves_unknown_coverage() -> None:
    summary = deck_summary_from_dto(
        {"deck": [{"id": "STRIKE"}, {"id": "UNKNOWN"}]},
        _extractor(),
        on_unknown_card="skip",
    )

    assert summary.deck_size == 2
    assert summary.unknown_card_count == 1
    assert summary.known_card_ratio == pytest.approx(0.5)


def test_unknown_card_raises_by_default() -> None:
    with pytest.raises(KeyError):
        deck_summary_from_dto({"deck": [{"id": "UNKNOWN"}]}, _extractor())


def test_state_kind_precedence_includes_event_boundary() -> None:
    dto = {"currentRoomType": "Shop", "boundary": "dto_boundary"}
    assert state_kind_from_dto(dto, event_boundary="event_boundary") == "Shop"
    assert state_kind_from_dto({"boundary": "dto_boundary"}, event_boundary="event_boundary") == (
        "event_boundary"
    )
    assert state_kind_from_dto({"boundary": "dto_boundary"}) == "dto_boundary"
    assert state_kind_from_dto({}, event_boundary="event_boundary") == "event_boundary"
    assert state_kind_from_dto({}) is None


def test_board_context_includes_state_kind() -> None:
    context = board_context_from_dto(
        {
            "hp": 42,
            "maxHp": 80,
            "gold": 99,
            "actFloor": 5,
            "totalFloor": 51,
            "currentRoomType": "RestSite",
        },
        event_boundary="map_select",
    )

    assert context == {
        "hp": 42,
        "max_hp": 80,
        "gold": 99,
        "act_floor": 5,
        "total_floor": 51,
        "state_kind": "RestSite",
    }
