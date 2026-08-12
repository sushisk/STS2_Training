"""Post-processing for God Mode-collected self-play logs.

See `Outputs/reports/god_mode_data_collection_proposal_20260812.md` (STS2_RL side)
and `docs/god_mode_data_collection_proposal_20260812.md` (this repo). God Mode
injects StrengthPower/BufferPower/RegenPower at ~10^9 stacks (Emulator's
`ApplyGodMode`).

This module performs a *contamination scrub* for training features: it intentionally
masks those three power IDs from every `playerPowers` list. It is not exact game-state
reconstruction. If ordinary gameplay added to or consumed stacks on the same merged
power entry, that information is intentionally discarded together with the God Mode
artifact. Consumers that need exact reconstructed power state must use a different,
baseline-aware representation instead of this scrub.

To avoid corrupting ordinary data during partial rollouts, a file is scrubbed only
when it contains exactly one `self_play_run_result`, that record is the final non-empty
record, and it proves both that God Mode was requested (`god_mode: true`) and that the
Emulator actually reported it active (`final_dto.godMode: true`). This matches the
self-play contract of one run per JSONL file and fails closed on appended/mixed logs.

`hp`/`maxHp` correction (the player is pinned near max HP throughout) is a separate,
still-open design question and is intentionally not attempted here - see the proposal
docs' "explicitly out of scope" sections.

CLI use::

    python -m sts2_training.god_mode_correction \\
        --input-dir data/self_play/godmode --output-dir data/self_play/godmode_corrected
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "GOD_MODE_POWER_IDS",
    "GodModeFlagMissingError",
    "correct_jsonl_file",
    "strip_god_mode_powers",
]

# Emulator power ids injected by GameInstance.ApplyGodMode (StrengthPower, BufferPower,
# RegenPower). This scrub masks the complete feature entries for these ids; it does not
# attempt to recover ordinary stacks that may have merged into the same entries.
GOD_MODE_POWER_IDS = frozenset({"STRENGTH_POWER", "BUFFER_POWER", "REGEN_POWER"})


class GodModeFlagMissingError(ValueError):
    """Raised when a file lacks unambiguous evidence that God Mode was active.

    The one-run-per-file contract requires exactly one terminal `self_play_run_result`,
    as the final non-empty record, with both top-level `god_mode: true` (requested) and
    `final_dto.godMode: true` (observed). Requiring all of these makes the scrub fail
    closed on partial rollouts and on files that were later appended with another run.
    """


def strip_god_mode_powers(node: Any) -> Any:
    """Return a deep copy with the three God Mode-contaminated power features masked.

    This is deliberately a feature scrub, not state reconstruction: every
    `playerPowers` entry whose id is in `GOD_MODE_POWER_IDS` is removed even if normal
    gameplay may also have changed the merged stack amount. Recursion mirrors
    `API/masking.py`'s recursive-scrub style because `playerPowers` can appear both
    under selection records and under a terminal `final_dto`.
    """
    if isinstance(node, dict):
        result: dict[str, Any] = {}
        for key, value in node.items():
            if key == "playerPowers" and isinstance(value, list):
                result[key] = [
                    strip_god_mode_powers(power)
                    for power in value
                    if not (isinstance(power, dict) and power.get("id") in GOD_MODE_POWER_IDS)
                ]
                continue
            result[key] = strip_god_mode_powers(value)
        return result
    if isinstance(node, list):
        return [strip_god_mode_powers(item) for item in node]
    return node


def _is_run_result(record: Any) -> bool:
    return isinstance(record, dict) and record.get("event") == "self_play_run_result"


def _is_verified_god_mode_run_result(record: Any) -> bool:
    if not (_is_run_result(record) and record.get("god_mode") is True):
        return False
    final_dto = record.get("final_dto")
    return isinstance(final_dto, dict) and final_dto.get("godMode") is True


def correct_jsonl_file(input_path: Path, output_path: Path) -> int:
    """Scrub one verified one-run God Mode JSONL log and return records written.

    The historical function/module name says "correct", but the operation is an
    intentional contamination scrub of three training features, not exact power-state
    reconstruction. Never mutates `input_path`; creates `output_path`'s parent
    directory if needed.
    """
    lines = [line for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    run_results = [record for record in records if _is_run_result(record)]
    verified_terminal = (
        len(run_results) == 1
        and bool(records)
        and records[-1] is run_results[0]
        and _is_verified_god_mode_run_result(records[-1])
    )
    if not verified_terminal:
        raise GodModeFlagMissingError(
            f"{input_path}: expected exactly one final self_play_run_result with both "
            "god_mode=true and final_dto.godMode=true - refusing to scrub a mixed, "
            "appended, or unverified log"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(strip_god_mode_powers(record), ensure_ascii=False))
            fh.write("\n")
    return len(records)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    input_paths = sorted(args.input_dir.glob("*.jsonl"))
    skipped: list[str] = []
    corrected = 0
    for input_path in input_paths:
        output_path = args.output_dir / input_path.name
        try:
            correct_jsonl_file(input_path, output_path)
        except GodModeFlagMissingError as exc:
            skipped.append(str(exc))
            continue
        corrected += 1

    print(
        json.dumps(
            {"files_found": len(input_paths), "files_corrected": corrected, "files_skipped": skipped},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
