from __future__ import annotations

import pytest

from sts2_training.api.contract import ApiContract, ApiProtocolError


def _response(masked: dict) -> dict:
    return {
        "decision_point_id": "d-terminal",
        "masked_emulator_dto": masked,
    }


@pytest.mark.parametrize("marker", ["terminal", "run_terminal"])
@pytest.mark.parametrize("outcome", [None, "", "draw", 1, True])
def test_terminal_decision_requires_valid_outcome(marker: str, outcome: object) -> None:
    contract = ApiContract(client_session_id="session-a")
    masked = {marker: True}
    if marker == "terminal":
        masked["legal_actions"] = []
    if outcome is not None:
        masked["outcome"] = outcome

    with pytest.raises(ApiProtocolError, match="outcome"):
        contract._validate_decision_payload(_response(masked))  # noqa: SLF001


@pytest.mark.parametrize(
    "masked",
    [
        {"terminal": True, "outcome": "victory", "legal_actions": []},
        {"terminal": True, "outcome": "defeat", "legal_actions": []},
        {"run_terminal": True, "outcome": "victory"},
        {
            "run_terminal": True,
            "outcome": "defeat",
            "boundary": "run_terminal",
            "legal_actions": [],
            "room_context": {},
            "history": [],
        },
    ],
)
def test_terminal_decision_accepts_victory_or_defeat(masked: dict) -> None:
    contract = ApiContract(client_session_id="session-a")

    contract._validate_decision_payload(_response(masked))  # noqa: SLF001
