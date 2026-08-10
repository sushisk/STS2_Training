from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Protocol

from sts2_training.selection_log import JsonlSelectionLogger


class RunEventLogger(Protocol):
    """Sink for the complete event stream that describes one run.

    Selection audit records and run-level completion records intentionally share this
    contract so a single JSONL stream can be replayed without consulting runner state.
    """

    def __call__(self, event: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...


class JsonlRunEventLogger(JsonlSelectionLogger):
    """Run-log name for the backward-compatible JSONL event writer.

    ``JsonlSelectionLogger`` remains available for existing callers. The runner uses
    this name to make it explicit that the same stream now also contains run-level
    events such as ``run_result``.
    """


class RunResultLike(Protocol):
    instance_id: str
    decisions_made: int
    final_dto: Mapping[str, Any]
    elapsed_s: float


def run_result_event(result: RunResultLike) -> dict[str, Any]:
    """Build the terminal event shared by run-log-producing entry points."""
    final_dto = deepcopy(dict(result.final_dto))
    return {
        "event": "run_result",
        "instance_id": result.instance_id,
        "decisions_made": result.decisions_made,
        "elapsed_s": result.elapsed_s,
        "outcome": final_dto.get("outcome"),
        "final_dto": final_dto,
    }


__all__ = ["JsonlRunEventLogger", "RunEventLogger", "run_result_event"]
