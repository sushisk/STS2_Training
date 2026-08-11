"""Combat decision selection logic (initial simple-logic stage)."""

from sts2_training.selection.action_classification import (
    CARD_ACTION_TYPE,
    CHOICE_CARD_ACTION_TYPE,
    CHOICE_CONFIRM_ACTION_TYPE,
    CHOICE_SKIP_ACTION_TYPE,
    available_actions,
    card_actions,
    choice_card_actions,
    choice_confirm_actions,
    choice_skip_actions,
)
from sts2_training.selection.choice_semantics import (
    CHOICE_OPERATIONS,
    CHOICE_SEMANTICS_VERSION,
    ChoiceCardSemantics,
    PendingChoiceContext,
    choice_option_id,
    parse_choice_semantics,
    pending_choice_context,
)
from sts2_training.selection.heuristic_selector import (
    HeuristicCombatSelector,
    NoAvailableActionError,
)

__all__ = [
    "CARD_ACTION_TYPE",
    "CHOICE_CARD_ACTION_TYPE",
    "CHOICE_CONFIRM_ACTION_TYPE",
    "CHOICE_OPERATIONS",
    "CHOICE_SEMANTICS_VERSION",
    "CHOICE_SKIP_ACTION_TYPE",
    "ChoiceCardSemantics",
    "HeuristicCombatSelector",
    "NoAvailableActionError",
    "PendingChoiceContext",
    "available_actions",
    "card_actions",
    "choice_card_actions",
    "choice_confirm_actions",
    "choice_option_id",
    "choice_skip_actions",
    "parse_choice_semantics",
    "pending_choice_context",
]
