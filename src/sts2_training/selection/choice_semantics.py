"""Policy-neutral consumer view of canonical pending card-choice semantics.

The Emulator owns the mechanic and RL owns masking/normalization. Training only consumes
that explicit public descriptor; it must never infer mechanics from prompt text, selector
names, card IDs, labels, or incidental option fields.
"""

from __future__ import annotations

import re
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

_OPAQUE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


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
    """Choice semantics plus decision-local option identity published by RL.

    ``identity_valid`` is deliberately separate from semantic parsing. A malformed token
    must never be silently indistinguishable from a genuinely absent selection because
    downstream policy uses this flag to fail closed.
    """

    semantics: ChoiceCardSemantics
    source_effect_id: str | None
    selected_option_ids: tuple[str, ...]
    option_ids: tuple[str, ...]
    identity_valid: bool


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
    reconstructed heuristically. Decision-local option IDs remain opaque strings. Identity
    parsing is fail-closed: malformed or duplicate IDs, selected-count mismatches, and
    selected/remaining overlap are exposed through ``identity_valid=False``.
    """
    pending = masked_emulator_dto.get("pendingChoice")
    if not isinstance(pending, Mapping):
        return None

    semantics = parse_choice_semantics(pending.get("choiceSemantics"))
    source_effect_id = _token_or_none(pending.get("sourceEffectId"))
    if not semantics.is_known:
        source_effect_id = None

    selected_option_ids, selected_valid = _token_sequence(pending.get("selectedOptionIds"))
    option_ids, options_valid = _option_id_sequence(pending.get("options"))

    selected_count = pending.get("selectedCount")
    selected_count_valid = (
        isinstance(selected_count, int)
        and not isinstance(selected_count, bool)
        and selected_count >= 0
        and selected_count == len(selected_option_ids)
    )

    identity_valid = (
        selected_valid
        and options_valid
        and selected_count_valid
        and len(set(selected_option_ids)) == len(selected_option_ids)
        and len(set(option_ids)) == len(option_ids)
        and set(selected_option_ids).isdisjoint(option_ids)
    )

    return PendingChoiceContext(
        semantics=semantics,
        source_effect_id=source_effect_id,
        selected_option_ids=selected_option_ids,
        option_ids=option_ids,
        identity_valid=identity_valid,
    )


def choice_option_id(action: JsonObject) -> str | None:
    """Return a choice-card action's opaque ID from canonical parameters only."""
    if action.get("action_type") != "choice_card":
        return None
    parameters = action.get("parameters")
    if not isinstance(parameters, Mapping):
        return None
    return _token_or_none(parameters.get("optionId"))


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
    if not isinstance(value, str):
        return None
    return value if _OPAQUE_TOKEN_RE.fullmatch(value) else None


def _token_sequence(value: Any) -> tuple[tuple[str, ...], bool]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return (), False
    tokens: list[str] = []
    valid = True
    for item in value:
        token = _token_or_none(item)
        if token is None:
            valid = False
            continue
        tokens.append(token)
    return tuple(tokens), valid


def _option_id_sequence(value: Any) -> tuple[tuple[str, ...], bool]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return (), False
    option_ids: list[str] = []
    valid = True
    for option in value:
        if not isinstance(option, Mapping):
            valid = False
            continue
        option_id = _token_or_none(option.get("optionId"))
        if option_id is None:
            valid = False
            continue
        option_ids.append(option_id)
    return tuple(option_ids), valid
