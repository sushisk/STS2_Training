"""Safety filter for ``choice_event_option`` (Whole Run event choice) decisions.

Unlike `room_heuristic`/`choice_card_heuristic`, this is not a soft quality preference -
it is a hard "never voluntarily choose a confirmed-lethal option while a safer one
exists" constraint, matching the Emulator's own client-side warning semantics
(`EventOption.WillKillPlayer`/`ThatDoesDamage`/`ThatDecreasesMaxHp`, evaluated and
exposed per-option as ``parameters.willKillPlayer`` by `GameInstance.BuildEventLegalActions`
- see that method's own comment for the exposure precedent). This module deliberately
does not attempt to rank event options by anything beyond that flag: unlike card/room
choices, there is no generic quality signal available here (an option's real-world
value is entirely event-specific and not safely inferable from its raw text key), so
"safe" candidates are chosen from uniformly at random by the caller.
"""

from __future__ import annotations

from collections.abc import Sequence

from sts2_training.selection.action_classification import (
    JsonObject,
    choice_event_option_actions,
)


def safe_event_option_candidates(legal_actions: Sequence[JsonObject]) -> list[JsonObject]:
    """Event-option candidates with confirmed-lethal ones (``willKillPlayer is True``)
    excluded, unless every candidate is confirmed-lethal (a genuinely forced death must
    still return something selectable). Options where the Emulator never evaluated the
    flag (``None``/missing - no ``ThatDoesDamage``-style call at all) are treated as not
    confirmed-lethal, not as safe; they simply carry no lethality claim either way.

    Returns an empty list when there are no ``choice_event_option`` candidates at all.
    """

    event_actions = choice_event_option_actions(legal_actions)
    if not event_actions:
        return []

    safe = [action for action in event_actions if not _is_confirmed_lethal(action)]
    return safe if safe else event_actions


def _is_confirmed_lethal(action: JsonObject) -> bool:
    params = action.get("parameters")
    if not isinstance(params, dict):
        return False
    return params.get("willKillPlayer") is True
