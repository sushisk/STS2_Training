"""Run-state board evaluation features and value-model interfaces."""

from sts2_training.board_eval.card_features import (
    DEFAULT_CARD_FEATURES_CSV,
    FEATURE_NAMES,
    CardFeatureExtractor,
    CardFeatures,
    ReferenceScaling,
    ScalingSource,
    Scope,
    UnknownCardError,
)
from sts2_training.board_eval.deck_summary import DECK_SUMMARY_FEATURE_NAMES, DeckSummary, summarize_deck
from sts2_training.board_eval.dto_adapter import (
    DeckCardRef,
    board_context_from_dto,
    deck_card_refs_from_dto,
    deck_features_from_dto,
    deck_features_with_unknown_count_from_dto,
    deck_summary_from_dto,
    state_kind_from_dto,
)
from sts2_training.board_eval.linear_value_function import LinearRunStateValueModel
from sts2_training.board_eval.run_state_value import RunStateValueModel
from sts2_training.board_eval.training_data import (
    MODEL_FEATURE_NAMES,
    NON_DECK_FEATURE_NAMES,
    NON_DECK_VALUE_FEATURE_NAMES,
    BoardStateExample,
    build_examples_from_log,
    iter_log_events,
    label_from_events,
)

__all__ = [
    "DECK_SUMMARY_FEATURE_NAMES",
    "DEFAULT_CARD_FEATURES_CSV",
    "FEATURE_NAMES",
    "MODEL_FEATURE_NAMES",
    "NON_DECK_FEATURE_NAMES",
    "NON_DECK_VALUE_FEATURE_NAMES",
    "BoardStateExample",
    "CardFeatureExtractor",
    "CardFeatures",
    "DeckCardRef",
    "DeckSummary",
    "LinearRunStateValueModel",
    "ReferenceScaling",
    "RunStateValueModel",
    "ScalingSource",
    "Scope",
    "UnknownCardError",
    "board_context_from_dto",
    "build_examples_from_log",
    "deck_card_refs_from_dto",
    "deck_features_from_dto",
    "deck_features_with_unknown_count_from_dto",
    "deck_summary_from_dto",
    "iter_log_events",
    "label_from_events",
    "state_kind_from_dto",
    "summarize_deck",
]
