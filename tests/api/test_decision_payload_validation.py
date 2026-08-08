from __future__ import annotations

import unittest

from sts2_training.api.contract import ApiContract, ApiProtocolError


class DecisionPayloadValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = ApiContract(client_session_id="session-a")

    @staticmethod
    def _response(masked: dict) -> dict:
        return {
            "decision_point_id": "decision-1",
            "masked_emulator_dto": masked,
        }

    def test_valid_legal_actions_are_accepted(self) -> None:
        self.contract._validate_decision_payload(
            self._response(
                {
                    "legal_actions": [
                        {
                            "action_id": "a-1",
                            "action_type": "card",
                            "is_available": True,
                            "parameters": {"cardId": "STRIKE"},
                        }
                    ]
                }
            )
        )

    def test_legacy_minimal_action_fixture_remains_accepted(self) -> None:
        self.contract._validate_decision_payload(
            self._response({"legal_actions": [{"action_id": "a-1"}]})
        )

    def test_run_terminal_without_actions_accepts_outcome(self) -> None:
        self.contract._validate_decision_payload(
            self._response({"run_terminal": True, "outcome": "victory"})
        )

    def test_run_terminal_without_outcome_is_rejected(self) -> None:
        with self.assertRaisesRegex(ApiProtocolError, "outcome"):
            self.contract._validate_decision_payload(
                self._response({"run_terminal": True})
            )

    def test_non_list_or_non_mapping_legal_actions_are_rejected(self) -> None:
        for legal_actions in ({"action_id": "a-1"}, ["a-1"]):
            with self.subTest(legal_actions=legal_actions):
                with self.assertRaises(ApiProtocolError):
                    self.contract._validate_decision_payload(
                        self._response({"legal_actions": legal_actions})
                    )

    def test_missing_empty_or_duplicate_action_ids_are_rejected(self) -> None:
        cases = [
            [{}],
            [{"action_id": ""}],
            [{"action_id": "a-1"}, {"action_id": "a-1"}],
        ]
        for legal_actions in cases:
            with self.subTest(legal_actions=legal_actions):
                with self.assertRaises(ApiProtocolError):
                    self.contract._validate_decision_payload(
                        self._response({"legal_actions": legal_actions})
                    )

    def test_invalid_optional_action_fields_are_rejected(self) -> None:
        cases = [
            {"action_id": "a-1", "action_type": ""},
            {"action_id": "a-1", "is_available": "yes"},
            {"action_id": "a-1", "parameters": []},
        ]
        for action in cases:
            with self.subTest(action=action):
                with self.assertRaises(ApiProtocolError):
                    self.contract._validate_decision_payload(
                        self._response({"legal_actions": [action]})
                    )

    def test_missing_legal_actions_is_rejected_for_nonterminal_decision(self) -> None:
        with self.assertRaisesRegex(ApiProtocolError, "legal_actions"):
            self.contract._validate_decision_payload(self._response({}))


if __name__ == "__main__":
    unittest.main()
