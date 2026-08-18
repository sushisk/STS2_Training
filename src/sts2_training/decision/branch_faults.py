"""Shared classification for branch faults returned by RL ``emulate_actions``.

Keep this module deliberately small and transport-shaped: both Beam Search policy and
Oracle JSONL persistence must assign the same stable signature without importing one
another. Unknown faults return ``None`` so callers can preserve their existing generic
retry/logging behavior.
"""

from __future__ import annotations

from typing import Any

SNAPSHOT_RESTORE_FAULT_SIGNATURE = "snapshot_restore_fault"
SETTLEMENT_TIMEOUT_SIGNATURE = "settlement_timeout"

_STRUCTURAL_FAULT_KINDS = frozenset(
    {
        "reference_integrity",
        "snapshot_restore",
        # Current STS2_RL detects the restored-snapshot monster Move gap before calling
        # Emulator Step(), avoiding the historical opaque settlement timeout.
        "snapshot_restore_missing_monster_move",
    }
)
_STRUCTURAL_DETAIL_MARKERS = (
    "snapshot restore rejected",
    "reference_integrity:",
    "dangling reference",
    "missing captured instance",
)
_SETTLEMENT_TIMEOUT_MARKER = "timed out waiting for the next decision point or settlement"


def classify_known_branch_fault(
    *,
    status: Any,
    fault_kind: Any,
    detail: Any,
) -> str | None:
    """Return a stable policy signature for the explicitly understood fault classes."""

    if status != "faulted":
        return None
    normalized_kind = fault_kind.lower() if isinstance(fault_kind, str) else ""
    normalized_detail = detail.lower() if isinstance(detail, str) else ""
    if normalized_kind in _STRUCTURAL_FAULT_KINDS or any(
        marker in normalized_detail for marker in _STRUCTURAL_DETAIL_MARKERS
    ):
        return SNAPSHOT_RESTORE_FAULT_SIGNATURE
    if _SETTLEMENT_TIMEOUT_MARKER in normalized_detail:
        return SETTLEMENT_TIMEOUT_SIGNATURE
    return None
