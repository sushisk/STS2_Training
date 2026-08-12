from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts2_training.god_mode_correction import (
    GodModeFlagMissingError,
    correct_jsonl_file,
    main,
    strip_god_mode_powers,
)

_GOD_MODE_POWERS = [
    {"id": "STRENGTH_POWER", "amount": 999999999},
    {"id": "BUFFER_POWER", "amount": 999999999},
    {"id": "REGEN_POWER", "amount": 999999999},
]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class StripGodModePowersTest(unittest.TestCase):
    def test_removes_only_the_three_god_mode_powers(self) -> None:
        dto = {
            "hp": 80,
            "playerPowers": [
                *_GOD_MODE_POWERS,
                {"id": "VULNERABLE_POWER", "amount": 2},
            ],
        }

        corrected = strip_god_mode_powers(dto)

        self.assertEqual(corrected["playerPowers"], [{"id": "VULNERABLE_POWER", "amount": 2}])
        self.assertEqual(corrected["hp"], 80)

    def test_does_not_mutate_input(self) -> None:
        dto = {"playerPowers": list(_GOD_MODE_POWERS)}

        strip_god_mode_powers(dto)

        self.assertEqual(len(dto["playerPowers"]), 3)

    def test_recurses_into_nested_dtos_at_any_depth(self) -> None:
        record = {
            "received": {"masked_emulator_dto": {"playerPowers": list(_GOD_MODE_POWERS)}},
            "final_dto": {"playerPowers": list(_GOD_MODE_POWERS)},
        }

        corrected = strip_god_mode_powers(record)

        self.assertEqual(corrected["received"]["masked_emulator_dto"]["playerPowers"], [])
        self.assertEqual(corrected["final_dto"]["playerPowers"], [])

    def test_leaves_records_without_player_powers_untouched(self) -> None:
        record = {"event": "selection", "received": {"masked_emulator_dto": {"boundary": "map_select"}}}

        corrected = strip_god_mode_powers(record)

        self.assertEqual(corrected, record)


class CorrectJsonlFileTest(unittest.TestCase):
    def test_refuses_a_file_without_a_god_mode_run_result_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "in.jsonl"
            output_path = Path(tmp) / "out.jsonl"
            _write_jsonl(
                input_path,
                [
                    {"event": "selection", "received": {"masked_emulator_dto": {}}},
                    {"event": "self_play_run_result", "god_mode": False},
                ],
            )

            with self.assertRaises(GodModeFlagMissingError):
                correct_jsonl_file(input_path, output_path)
            self.assertFalse(output_path.exists())

    def test_corrects_every_record_and_preserves_record_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "in.jsonl"
            output_path = Path(tmp) / "nested" / "out.jsonl"
            _write_jsonl(
                input_path,
                [
                    {
                        "event": "selection",
                        "received": {"masked_emulator_dto": {"playerPowers": list(_GOD_MODE_POWERS)}},
                    },
                    {
                        "event": "self_play_run_result",
                        "god_mode": True,
                        "final_dto": {"playerPowers": list(_GOD_MODE_POWERS)},
                    },
                ],
            )

            count = correct_jsonl_file(input_path, output_path)

            self.assertEqual(count, 2)
            written = _read_jsonl(output_path)
            self.assertEqual(len(written), 2)
            self.assertEqual(written[0]["received"]["masked_emulator_dto"]["playerPowers"], [])
            self.assertEqual(written[1]["final_dto"]["playerPowers"], [])

    def test_never_mutates_the_input_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "in.jsonl"
            output_path = Path(tmp) / "out.jsonl"
            _write_jsonl(
                input_path,
                [{"event": "self_play_run_result", "god_mode": True, "final_dto": {"playerPowers": list(_GOD_MODE_POWERS)}}],
            )
            original_text = input_path.read_text(encoding="utf-8")

            correct_jsonl_file(input_path, output_path)

            self.assertEqual(input_path.read_text(encoding="utf-8"), original_text)


class MainCliTest(unittest.TestCase):
    def test_processes_all_files_and_reports_skips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "in"
            output_dir = Path(tmp) / "out"
            input_dir.mkdir()

            _write_jsonl(
                input_dir / "good.jsonl",
                [{"event": "self_play_run_result", "god_mode": True, "final_dto": {}}],
            )
            _write_jsonl(
                input_dir / "bad.jsonl",
                [{"event": "self_play_run_result", "god_mode": False, "final_dto": {}}],
            )

            code = main(["--input-dir", str(input_dir), "--output-dir", str(output_dir)])

            self.assertEqual(code, 1)
            self.assertTrue((output_dir / "good.jsonl").exists())
            self.assertFalse((output_dir / "bad.jsonl").exists())

    def test_exits_zero_when_nothing_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "in"
            output_dir = Path(tmp) / "out"
            input_dir.mkdir()
            _write_jsonl(
                input_dir / "good.jsonl",
                [{"event": "self_play_run_result", "god_mode": True, "final_dto": {}}],
            )

            code = main(["--input-dir", str(input_dir), "--output-dir", str(output_dir)])

            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
