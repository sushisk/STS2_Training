"""Adapt masked emulator DTOs into board_eval's Run-state feature pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from sts2_training.board_eval.card_features import CardFeatureExtractor, CardFeatures, UnknownCardError
from sts2_training.board_eval.deck_summary import DeckSummary, summarize_deck

UnknownCardPolicy = Literal["skip", "raise"]


@dataclass(frozen=True)
class DeckCardRef:
    card_id: str
    upgraded: bool


def deck_card_refs_from_dto(dto: Mapping[str, object]) -> list[DeckCardRef]:
    deck = dto.get("deck")
    if deck is None:
        return []
    if not isinstance(deck, Sequence) or isinstance(deck, (str, bytes)):
        raise ValueError("masked_emulator_dto['deck'] must be a sequence when present")

    refs: list[DeckCardRef] = []
    for index, entry in enumerate(deck):
        if not isinstance(entry, Mapping):
            raise ValueError(f"masked_emulator_dto['deck'][{index}] must be a mapping")
        card_id = entry.get("id")
        if not isinstance(card_id, str) or not card_id:
            raise ValueError(f"masked_emulator_dto['deck'][{index}].id must be a non-empty string")
        upgraded = entry.get("upgraded", False)
        if not isinstance(upgraded, bool):
            raise ValueError(
                f"masked_emulator_dto['deck'][{index}].upgraded must be a boolean when present"
            )
        refs.append(DeckCardRef(card_id, upgraded))
    return refs


def deck_features_with_unknown_count_from_dto(
    dto: Mapping[str, object],
    extractor: CardFeatureExtractor,
    *,
    on_unknown_card: UnknownCardPolicy = "raise",
) -> tuple[list[CardFeatures], int]:
    if on_unknown_card not in ("raise", "skip"):
        raise ValueError(f"unknown on_unknown_card policy: {on_unknown_card!r}")

    cards: list[CardFeatures] = []
    unknown_count = 0
    for ref in deck_card_refs_from_dto(dto):
        try:
            cards.append(extractor.extract(ref.card_id, upgraded=ref.upgraded))
        except UnknownCardError:
            if on_unknown_card == "raise":
                raise
            unknown_count += 1
    return cards, unknown_count


def deck_features_from_dto(
    dto: Mapping[str, object],
    extractor: CardFeatureExtractor,
    *,
    on_unknown_card: UnknownCardPolicy = "raise",
) -> list[CardFeatures]:
    cards, _ = deck_features_with_unknown_count_from_dto(
        dto,
        extractor,
        on_unknown_card=on_unknown_card,
    )
    return cards


def deck_summary_from_dto(
    dto: Mapping[str, object],
    extractor: CardFeatureExtractor,
    *,
    on_unknown_card: UnknownCardPolicy = "raise",
) -> DeckSummary:
    cards, unknown_count = deck_features_with_unknown_count_from_dto(
        dto,
        extractor,
        on_unknown_card=on_unknown_card,
    )
    return summarize_deck(cards, unknown_card_count=unknown_count)


def state_kind_from_dto(
    dto: Mapping[str, object],
    *,
    event_boundary: object | None = None,
) -> str | None:
    """Return an available Whole-Run decision-state discriminator without fixing policy yet."""
    room_type = dto.get("currentRoomType")
    if isinstance(room_type, str) and room_type:
        return room_type
    if isinstance(event_boundary, str) and event_boundary:
        return event_boundary
    boundary = dto.get("boundary")
    if isinstance(boundary, str) and boundary:
        return boundary
    return None


def board_context_from_dto(
    dto: Mapping[str, object],
    *,
    event_boundary: object | None = None,
) -> dict[str, object]:
    return {
        "hp": dto.get("hp"),
        "max_hp": dto.get("maxHp"),
        "gold": dto.get("gold"),
        "act_floor": dto.get("actFloor"),
        "total_floor": dto.get("totalFloor"),
        "state_kind": state_kind_from_dto(dto, event_boundary=event_boundary),
    }
