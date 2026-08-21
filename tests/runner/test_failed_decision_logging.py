"""A decision that raises must still leave its search trace behind.

Detailed logging ran after `decide()` returned, so the one decision whose trace explains
why a run died was the one decision never written. Two Whole Run evaluations aborted with
`AllBranchesFaultedError` and neither log contained the search that raised: `search_start`
and `search_end` counts matched exactly, because the failing search contributed neither.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.decision.combat_decision import COMBAT_BEAM_ACTION_TYPES
from sts2_training.decision.engine import CombatDecisionEngine, DecisionOutcome
from sts2_training.decision.search_trace import SearchTraceStart
from sts2_training.runner.floor_reach_eval import (
    _RunState,
    _TrackingCombatDecisionEngine,
)


class _Boom(RuntimeError):
    pass


_ROOT_DTO = {
    "boundary": "stable",
    "turnNumber": 3,
    "totalFloor": 11,
    "legal_actions": [
        {"action_id": "0", "action_type": "system", "label": "End Turn", "is_available": True},
    ],
}
_DECISION = {"decision_point_id": "d-root-42", "masked_emulator_dto": _ROOT_DTO}


class _Client:
    instance_type = "whole_run"
    max_emulate_actions_items = 8
    pending_retry = None
    session_invalid = False


def _search_start() -> SearchTraceStart:
    return SearchTraceStart(
        search_id="s-1",
        instance_id="inst",
        root_decision_point_id="d-root-42",
        beam_width=8,
        top_k_actions=4,
        max_depth=2,
        max_continuation_steps=8,
        time_budget_ms=None,
        pruner_name="stub",
        pruner_version="stub",
    )


class FailedDecisionLoggingTest(unittest.IsolatedAsyncioTestCase):
    async def _run(self, *, decision, trace_first: bool = True) -> list[dict]:
        records: list[dict] = []
        engine = _TrackingCombatDecisionEngine(
            _Client(),
            state=_RunState(),
            detailed_logger=records.append,
            beam_config=BeamSearchConfig(
                beam_searchable_action_types=COMBAT_BEAM_ACTION_TYPES,
            ),
        )
        collector = engine._score_trace_collector
        self.assertIsNotNone(collector)

        async def _raise(_self, instance_id, *, timeout_s, decision=None):
            del instance_id, timeout_s, decision
            if trace_first:
                collector.record(_search_start())
            raise _Boom("all emulate_actions branch results faulted")

        with (
            patch.object(CombatDecisionEngine, "decide", _raise),
            self.assertRaises(_Boom),
        ):
            await engine.decide("inst", timeout_s=5.0, decision=decision)
        return records

    async def test_the_trace_of_the_failing_search_is_written(self) -> None:
        records = await self._run(decision=_DECISION)

        traces = [r for r in records if r["event"] == "score_trace"]
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["decision_point_id"], "d-root-42")
        self.assertEqual(traces[0]["decision_source"], "failed")
        self.assertIsNone(traces[0]["selected_action_id"])
        self.assertEqual(traces[0]["trace_event"]["event_type"], "search_start")

    async def test_the_failure_itself_is_recorded_with_the_board(self) -> None:
        records = await self._run(decision=_DECISION)

        failures = [r for r in records if r["event"] == "decision_failed"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["error"]["type"], "_Boom")
        self.assertIn("faulted", failures[0]["error"]["message"])
        self.assertEqual(failures[0]["decision_point_id"], "d-root-42")
        self.assertEqual(failures[0]["masked_emulator_dto"]["totalFloor"], 11)

    async def test_the_decision_point_falls_back_to_the_trace(self) -> None:
        """`decide()` is also called with no decision in hand; the trace still names it."""
        records = await self._run(decision=None)

        failures = [r for r in records if r["event"] == "decision_failed"]
        self.assertEqual(failures[0]["decision_point_id"], "d-root-42")
        self.assertIsNone(failures[0]["masked_emulator_dto"])

    async def test_a_failure_before_any_trace_still_records_the_error(self) -> None:
        records = await self._run(decision=None, trace_first=False)

        failures = [r for r in records if r["event"] == "decision_failed"]
        self.assertEqual(len(failures), 1)
        self.assertIsNone(failures[0]["decision_point_id"])
        self.assertEqual([r for r in records if r["event"] == "score_trace"], [])


class SuccessfulDecisionLoggingTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_successful_decision_still_labels_its_trace(self) -> None:
        """The success path now passes the same fields explicitly; pin them."""
        records: list[dict] = []
        engine = _TrackingCombatDecisionEngine(
            _Client(),
            state=_RunState(),
            detailed_logger=records.append,
            beam_config=BeamSearchConfig(
                beam_searchable_action_types=COMBAT_BEAM_ACTION_TYPES,
            ),
        )
        collector = engine._score_trace_collector

        outcome = DecisionOutcome(
            decision=_DECISION,
            chosen_action_id="0",
            source="beam_search",
            beam_result=None,
        )

        async def _succeed(_self, instance_id, *, timeout_s, decision=None):
            del instance_id, timeout_s, decision
            collector.record(_search_start())
            return outcome

        with patch.object(CombatDecisionEngine, "decide", _succeed):
            await engine.decide("inst", timeout_s=5.0, decision=_DECISION)

        traces = [r for r in records if r["event"] == "score_trace"]
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["decision_point_id"], "d-root-42")
        self.assertEqual(traces[0]["decision_source"], "beam_search")
        self.assertEqual(traces[0]["selected_action_id"], "0")
        self.assertEqual([r for r in records if r["event"] == "decision_failed"], [])


if __name__ == "__main__":
    unittest.main()
