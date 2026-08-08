from __future__ import annotations

import pytest

from sts2_training.api.contract import ApiContract, ApiProtocolError


def _response(masked: dict) -> dict:
    return {
        "decision_point_id": "d-terminal",
        "masked_emulator_dto": masked,
    }


@pytest.mark.parametrize("marker", ["terminal", "run_terminal"])
def test_terminal_decision_requires_outcome(marker: str) -> None:
    contract = ApiContract(client_session_id="session-a")
    masked = {marker: True}
    if marker == "terminal":
        masked["legal_actions"] = []

    with pytest.raises(ApiProtocolError, match="outcome"):
        contract._validate_decision_payload(_response(masked))  # noqa: SLF001


@pytest.mark.parametrize("marker", ["terminal", "run_terminal"])
@pytest.mark.parametrize("outcome", [None, "", "draw", 1, True, [], {}])
def test_terminal_decision_rejects_invalid_outcome(marker: str, outcome: object) -> None:
    contract = ApiContract(client_session_id="session-a")
    masked = {marker: True, "outcome": outcome}
    if marker == "terminal":
        masked["legal_actions"] = []

    with pytest.raises(ApiProtocolError, match="outcome"):
        contract._validate_decision_payload(_response(masked))  # noqa: SLF001


@pytest.mark.parametrize("outcome", ["victory", "defeat", None, "in_progress"])
def test_non_terminal_decision_rejects_outcome_field(outcome: object) -> None:
    contract = ApiContract(client_session_id="session-a")

    with pytest.raises(ApiProtocolError, match="non-terminal.*outcome"):
        contract._validate_decision_payload(  # noqa: SLF001
            _response({"legal_actions": [], "outcome": outcome})
        )


@pytest.mark.parametrize("marker", ["terminal", "run_terminal"])
def test_terminal_decision_rejects_non_empty_legal_actions(marker: str) -> None:
    contract = ApiContract(client_session_id="session-a")
    masked = {
        marker: True,
        "outcome": "victory",
        "legal_actions": [{"action_id": "stale-action"}],
    }

    with pytest.raises(ApiProtocolError, match="legal_actions.*empty"):
        contract._validate_decision_payload(_response(masked))  # noqa: SLF001


def test_run_terminal_rejects_explicit_null_legal_actions() -> None:
    contract = ApiContract(client_session_id="session-a")

    with pytest.raises(ApiProtocolError, match="legal_actions.*list"):
        contract._validate_decision_payload(  # noqa: SLF001
            _response(
                {
                    "run_terminal": True,
                    "outcome": "victory",
                    "legal_actions": None,
                }
            )
        )


@pytest.mark.parametrize("marker", ["terminal", "run_terminal"])
@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_terminal_markers_must_be_boolean(marker: str, value: object) -> None:
    contract = ApiContract(client_session_id="session-a")
    masked = {marker: value, "legal_actions": []}

    with pytest.raises(ApiProtocolError, match=rf"{marker}.*boolean"):
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
