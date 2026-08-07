from __future__ import annotations

import unittest

from sts2_training.api.contract import ApiContract


class SimulationOptionsContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = ApiContract(client_session_id="session-a")
        self.contract._accept_start_instance(
            {"status": "completed", "instance_id": "inst-001"}
        )

    def _build(self, simulation_options) -> dict:
        return self.contract._build_emulate_action(
            1,
            "inst-001",
            "root",
            "branch-1",
            1,
            "decision-1",
            "action-1",
            simulation_options,
        )

    def test_invalid_limits_fail_before_request_is_built(self) -> None:
        for field in ("max_depth", "max_steps", "max_time_ms", "max_hypotheses"):
            for invalid in (0, -1, True, 1.5):
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "integer"):
                        self._build({field: invalid})

    def test_unsupported_stop_condition_fails_locally(self) -> None:
        with self.assertRaisesRegex(ValueError, "not supported"):
            self._build({"stop_condition": "never"})

    def test_valid_options_and_unknown_extension_keys_are_preserved(self) -> None:
        options = {
            "max_depth": 1,
            "max_steps": 2,
            "max_time_ms": 3,
            "max_hypotheses": 4,
            "stop_condition": "next_decision",
            "future_extension": {"enabled": True},
        }
        request = self._build(options)
        self.assertEqual(request["simulation_options"], options)

    def test_none_values_remain_accepted_for_optional_known_fields(self) -> None:
        options = {"max_depth": None, "stop_condition": None}
        request = self._build(options)
        self.assertEqual(request["simulation_options"], options)

    def test_non_mapping_options_are_rejected_explicitly(self) -> None:
        with self.assertRaisesRegex(TypeError, "mapping"):
            self._build([("max_depth", 1)])


if __name__ == "__main__":
    unittest.main()
