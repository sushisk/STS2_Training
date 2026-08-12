"""Post-processing for God Mode-collected self-play logs.

See `Outputs/reports/god_mode_data_collection_proposal_20260812.md` (STS2_RL side)
and `docs/god_mode_data_collection_proposal_20260812.md` (this repo) - God Mode
invincibility is applied via StrengthPower/BufferPower/RegenPower at ~10^9 stacks
(Emulator's `ApplyGodMode`). Every `playerPowers` entry for these three powers is
therefore a collection-method artifact, not real game state, and must be removed
before any downstream consumer treats it as a training feature.

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

# Emulator power ids applied by GameInstance.ApplyGodMode (StrengthPower, BufferPower,
# RegenPower) - confirmed against real collected JSONL logs' playerPowers.id values.
GOD_MODE_POWER_IDS = frozenset({"STRENGTH_POWER", "BUFFER_POWER", "REGEN_POWER"})


class GodModeFlagMissingError(ValueError):
    """Raised when a file is asked to be corrected but never claims to be God Mode data.

    A JSONL file with no `self_play_run_result` record carrying `god_mode: true` is
    refused outright rather than "corrected" - silently stripping powers from an
    ordinary run would corrupt real playerPowers data instead of an artifact.
    """


def strip_god_mode_powers(node: Any) -> Any:
    """Return a deep copy of `node` with god-mode `playerPowers` entries removed.

    Recurses through the whole structure (mirroring `API/masking.py`'s recursive-scrub
    style) rather than assuming one fixed nesting depth, since `playerPowers` can
    appear both under a selection record's `received.masked_emulator_dto` and under a
    `self_play_run_result` record's `final_dto`.
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


def _is_god_mode_run_result(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and record.get("event") == "self_play_run_result"
        and record.get("god_mode") is True
    )


def correct_jsonl_file(input_path: Path, output_path: Path) -> int:
    """Correct one God Mode JSONL log, refusing to process a file that isn't one.

    Returns the number of records written. Never mutates `input_path`; creates
    `output_path`'s parent directory if needed.
    """
    lines = [line for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    if not any(_is_god_mode_run_result(record) for record in records):
        raise GodModeFlagMissingError(
            f"{input_path}: no self_play_run_result record with god_mode=true - "
            "refusing to correct a file that may not be God Mode data"
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
