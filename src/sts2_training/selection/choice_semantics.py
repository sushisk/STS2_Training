"""Policy-neutral consumer view of canonical pending card-choice semantics.

The Emulator owns the mechanic and RL owns masking/normalization. Training only consumes
that explicit public descriptor; it must never infer mechanics from prompt text, selector
names, card IDs, labels, or incidental option fields.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

JsonObject = Mapping[str, Any]

CHOICE_SEMANTICS_VERSION = 1
CHOICE_OPERATIONS = frozenset(
    {
        "gain",
        "discard",
        "exhaust",
        "upgrade",
        "retrieve",
        "play",
        "replay",
        "remove",
        "transform",
        "unknown",
    }
)
CHOICE_EFFECTS = frozenset({"move", "modify", "play", "replace"})
CHOICE_ZONES = frozenset(
    {
        "hand",
        "draw",
        "draw_pile",
        "discard",
        "discard_pile",
        "exhaust",
        "exhaust_pile",
        "play",
        "play_pile",
        "deck",
        "master_deck",
        "reward",
        "generated",
        "none",
        "unknown",
    }
)
CHOICE_MODIFIERS = frozenset({"upgrade"})


@dataclass(frozen=True)
class ChoiceCardSemantics:
    """Canonical v1 mechanic descriptor after Training-side defensive parsing."""

    version: int
    operation: str
    effect: str | None = None
    source_zone: str | None = None
    destination_zone: str | None = None
    modifier: str | None = None
    order_matters: bool | None = None
    replacement_allowed: bool | None = None

    @property
    def is_known(self) -> bool:
        return self.operation != "unknown"


@dataclass(frozen=True)
class PendingChoiceContext:
    """Choice semantics plus decision-local option identity published by RL."""

    semantics: ChoiceCardSemantics
    source_effect_id: str | None
    selected_option_ids: tuple[str, ...]
    option_ids: tuple[str, ...]


_UNKNOWN = ChoiceCardSemantics(CHOICE_SEMANTICS_VERSION, "unknown")


def parse_choice_semantics(raw: Any) -> ChoiceCardSemantics:
    """Parse public v1 semantics; malformed/future descriptors become neutral unknown."""
    if not isinstance(raw, Mapping):
        return _UNKNOWN

    version = raw.get("version")
    operation = raw.get("operation")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != CHOICE_SEMANTICS_VERSION
        or not isinstance(operation, str)
        or operation not in CHOICE_OPERATIONS
        or operation == "unknown"
    ):
        return _UNKNOWN

    try:
        effect = _enum_or_none(raw.get("effect"), CHOICE_EFFECTS)
        source_zone = _enum_or_none(raw.get("sourceZone"), CHOICE_ZONES)
        destination_zone = _enum_or_none(raw.get("destinationZone"), CHOICE_ZONES)
        modifier = _enum_or_none(raw.get("modifier"), CHOICE_MODIFIERS)
        order_matters = _bool_or_none(raw.get("orderMatters"))
        replacement_allowed = _bool_or_none(raw.get("replacementAllowed"))
    except ValueError:
        return _UNKNOWN

    return ChoiceCardSemantics(
        version=CHOICE_SEMANTICS_VERSION,
        operation=operation,
        effect=effect,
        source_zone=source_zone,
        destination_zone=destination_zone,
        modifier=modifier,
        order_matters=order_matters,
        replacement_allowed=replacement_allowed,
    )


def pending_choice_context(masked_emulator_dto: JsonObject) -> PendingChoiceContext | None:
    """Return canonical pending-choice context when a public pendingChoice is present.

    Missing/old-producer semantics are represented as ``operation='unknown'`` rather than
    reconstructed heuristically. Decision-local option IDs remain opaque strings.
    """
    pending = masked_emulator_dto.get("pendingChoice")
    if not isinstance(pending, Mapping):
        return None

    semantics = parse_choice_semantics(pending.get("choiceSemantics"))
    source_effect_id = _token_or_none(pending.get("sourceEffectId"))
    if not semantics.is_known:
        source_effect_id = None

    selected_option_ids = _token_sequence(pending.get("selectedOptionIds"))
    option_ids: list[str] = []
    raw_options = pending.get("options")
    if isinstance(raw_options, Sequence) and not isinstance(raw_options, (str, bytes)):
        for option in raw_options:
            if isinstance(option, Mapping):
                option_id = _token_or_none(option.get("optionId"))
                if option_id is not None:
                    option_ids.append(option_id)

    return PendingChoiceContext(
        semantics=semantics,
        source_effect_id=source_effect_id,
        selected_option_ids=selected_option_ids,
        option_ids=tuple(option_ids),
    )


def choice_option_id(action: JsonObject) -> str | None:
    """Return the opaque decision-local option ID attached to a choice-card action."""
    return _token_or_none(action.get("optionId"))


def _enum_or_none(value: Any, allowed: frozenset[str]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in allowed:
        raise ValueError("invalid semantic enum")
    return value


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("invalid semantic boolean")
    return value


def _token_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and bool(value) else None


def _token_sequence(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(token for item in value if (token := _token_or_none(item)) is not None)
