from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import unittest
from pathlib import Path

from sts2_training.runner._cli import add_common_arguments, configure_logging, print_result
from sts2_training.runner.episode import EpisodeResult


class AddCommonArgumentsTest(unittest.TestCase):
    def test_log_level_defaults_to_warning(self) -> None:
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)

        args = parser.parse_args([])

        self.assertEqual(args.log_level, "WARNING")

    def test_log_level_accepts_known_values(self) -> None:
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)

        args = parser.parse_args(["--log-level", "DEBUG"])

        self.assertEqual(args.log_level, "DEBUG")

    def test_unknown_log_level_is_rejected(self) -> None:
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)

        with self.assertRaises(SystemExit):
            parser.parse_args(["--log-level", "NOISY"])

    def test_non_positive_max_decisions_is_rejected(self) -> None:
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)

        for value in ("0", "-1"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parser.parse_args(["--max-decisions", value])

    def test_non_positive_beam_depth_is_rejected(self) -> None:
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)

        for value in ("0", "-1"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parser.parse_args(["--beam-depth", value])


class ConfigureLoggingTest(unittest.TestCase):
    def tearDown(self) -> None:
        # basicConfig() only has an effect the first time it's called per
        # process, so undo it here to keep this test isolated from others.
        logging.root.handlers.clear()
        logging.root.setLevel(logging.WARNING)

    def test_root_logger_level_is_set(self) -> None:
        configure_logging("DEBUG")

        self.assertEqual(logging.root.level, logging.DEBUG)


class PrintResultTest(unittest.TestCase):
    def test_prints_expected_fields_as_json(self) -> None:
        import contextlib
        import io
        import json

        result = EpisodeResult(instance_id="inst-1", decisions_made=3, final_dto={"outcome": "victory"}, elapsed_s=1.5)
        buffer = io.StringIO()

        with contextlib.redirect_stdout(buffer):
            print_result(result)

        payload = json.loads(buffer.getvalue())
        self.assertEqual(
            payload,
            {
                "instance_id": "inst-1",
                "decisions_made": 3,
                "elapsed_s": 1.5,
                "final_dto": {"outcome": "victory"},
            },
        )


class ModuleCliTest(unittest.TestCase):
    def test_start_modules_do_not_warn_about_preimport(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        src_path = str(repo_root / "src")
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            src_path if not existing_pythonpath else src_path + os.pathsep + existing_pythonpath
        )

        modules = (
            "sts2_training.runner.start_combat_from_state",
            "sts2_training.runner.start_new_run",
            "sts2_training.runner.start_run_from_state",
        )
        for module in modules:
            with self.subTest(module=module):
                completed = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=repo_root,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertNotIn("found in sys.modules", completed.stderr)


if __name__ == "__main__":
    unittest.main()
