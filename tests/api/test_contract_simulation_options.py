from __future__ import annotations

import unittest

from sts2_training.api.contract import ApiContract


class SimulationOptionsContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = ApiContract(client_session_id="session-a")
        self.contract._accept_start_instance(
            {"status": "completed", "instance_id": "inst-001"}
        )

    def _build(self, options) -> dict:
        return self.contract._build_emulate_action(
            1, "inst-001", "root", "branch-1", 1, "decision-1", "action-1", options
        )

    def test_invalid_known_options_fail_locally(self) -> None:
        for field in ("max_depth", "max_steps", "max_time_ms", "max_hypotheses"):
            for value in (0, True, 1.5):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        self._build({field: value})
        with self.assertRaisesRegex(ValueError, "not supported"):
            self._build({"stop_condition": "never"})

    def test_valid_options_are_preserved(self) -> None:
        options = {
            "max_depth": 1,
            "max_steps": None,
            "stop_condition": "next_decision",
            "future_extension": True,
        }
        self.assertEqual(self._build(options)["simulation_options"], options)


if __name__ == "__main__":
    unittest.main()
