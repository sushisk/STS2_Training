"""Run-state value model interface, intentionally separate from combat scoring."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping


class RunStateValueModel(ABC):
    """Predict a Run-level win probability from a public Run-state mapping."""

    @abstractmethod
    def predict_win_probability(self, run_state: Mapping[str, object]) -> float:
        """Return the predicted probability that the current Run eventually wins."""
        raise NotImplementedError
