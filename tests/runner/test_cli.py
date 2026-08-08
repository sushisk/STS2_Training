from __future__ import annotations

import argparse
import logging
import unittest

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


if __name__ == "__main__":
    unittest.main()
