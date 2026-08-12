"""Coverage for ``best_event_option`` without a live server."""

from __future__ import annotations

import unittest

from sts2_training.api.transport import TransportError
from sts2_training.decision.event_branch_search import best_event_option

_OVERCOME = {"action_id": "a-overcome", "action_type": "choice_event_option", "parameters": {}}
_HOLD_ON = {"action_id": "a-hold-on", "action_type": "choice_event_option", "parameters": {}}
_CARD = {"action_id": "a-card", "action_type": "card", "parameters": {}}


class _FakeClient:
    def __init__(
        self,
        hp_by_action: dict[str, float | None],
        *,
        fail_action_id: str | None = None,
        transport_fail_action_id: str | None = None,
    ) -> None:
        self.hp_by_action = hp_by_action
        self.fail_action_id = fail_action_id
        self.transport_fail_action_id = transport_fail_action_id
        self.emulate_batches: list[list[dict[str, object]]] = []
        self.released: list[list[str]] = []
        self.release_timeouts: list[float] = []

    async def emulate_actions(
        self,
        instance_id,
        items,
        *,
        timeout_s,
        simulation_options=None,
    ):
        batch = [dict(item) for item in items]
        self.emulate_batches.append(batch)
        if any(item["action_id"] == self.transport_fail_action_id for item in batch):
            raise TransportError("candidate transport failed", completion_uncertain=True)

        branch_results = {}
        for item in batch:
            action_id = item["action_id"]
            branch_id = item["branch_id"]
            if action_id == self.fail_action_id:
                branch_results[branch_id] = {
                    "status": "faulted",
                    "error": "candidate simulation failed",
                }
                continue
            hp = self.hp_by_action.get(action_id)
            if hp is None:
                branch_results[branch_id] = {"status": "faulted"}
                continue
            branch_results[branch_id] = {
                "status": "completed",
                "masked_emulator_dto": {"hp": hp, "boundary": "map_select"},
            }
        return {"status": "completed", "branch_results": branch_results}

    async def release_branches(self, instance_id, branch_ids, *, timeout_s):
        self.released.append(list(branch_ids))
        self.release_timeouts.append(timeout_s)
        return {}


class BestEventOptionTest(unittest.IsolatedAsyncioTestCase):
    async def test_prefers_the_option_that_preserves_the_most_hp(self) -> None:
        client = _FakeClient({"a-overcome": 60.0, "a-hold-on": 53.0})

        result = await best_event_option(
            client,
            instance_id="inst-1",
            decision_point_id="d1",
            legal_actions=[_OVERCOME, _HOLD_ON],
            timeout_s=5.0,
        )

        self.assertEqual(result, "a-overcome")
        self.assertEqual(len(client.emulate_batches), 1)
        self.assertEqual(len(client.emulate_batches[0]), 2)
        branch_ids = {item["branch_id"] for item in client.emulate_batches[0]}
        self.assertEqual(set(client.released[0]), branch_ids)
        self.assertLess(client.release_timeouts[0], 5.0)

    async def test_uses_distinct_non_root_rng_hypotheses(self) -> None:
        client = _FakeClient({"a-overcome": 60.0, "a-hold-on": 53.0})

        await best_event_option(
            client,
            instance_id="inst-1",
            decision_point_id="d1",
            legal_actions=[_OVERCOME, _HOLD_ON],
            timeout_s=5.0,
        )

        rng_ids = [item["rng_id"] for item in client.emulate_batches[0]]
        self.assertEqual(rng_ids, [1, 2])
        self.assertNotIn(0, rng_ids)

    async def test_excludes_branches_that_resolve_to_death(self) -> None:
        client = _FakeClient({"a-overcome": 0.0, "a-hold-on": 3.0})

        result = await best_event_option(
            client,
            instance_id="inst-1",
            decision_point_id="d1",
            legal_actions=[_OVERCOME, _HOLD_ON],
            timeout_s=5.0,
        )

        self.assertEqual(result, "a-hold-on")

    async def test_ignores_non_finite_hp(self) -> None:
        client = _FakeClient({"a-overcome": float("nan"), "a-hold-on": 3.0})

        result = await best_event_option(
            client,
            instance_id="inst-1",
            decision_point_id="d1",
            legal_actions=[_OVERCOME, _HOLD_ON],
            timeout_s=5.0,
        )

        self.assertEqual(result, "a-hold-on")

    async def test_returns_none_when_every_branch_dies_or_fails(self) -> None:
        client = _FakeClient({"a-overcome": 0.0, "a-hold-on": None})

        result = await best_event_option(
            client,
            instance_id="inst-1",
            decision_point_id="d1",
            legal_actions=[_OVERCOME, _HOLD_ON],
            timeout_s=5.0,
        )

        self.assertIsNone(result)

    async def test_returns_none_with_zero_or_one_event_option_candidates(self) -> None:
        client = _FakeClient({})

        result_empty = await best_event_option(
            client,
            instance_id="i",
            decision_point_id="d",
            legal_actions=[_CARD],
            timeout_s=5.0,
        )
        result_one = await best_event_option(
            client,
            instance_id="i",
            decision_point_id="d",
            legal_actions=[_OVERCOME, _CARD],
            timeout_s=5.0,
        )

        self.assertIsNone(result_empty)
        self.assertIsNone(result_one)
        self.assertEqual(client.emulate_batches, [])

    async def test_one_candidate_fault_does_not_sink_the_others(self) -> None:
        client = _FakeClient(
            {"a-overcome": 60.0, "a-hold-on": 53.0},
            fail_action_id="a-hold-on",
        )

        result = await best_event_option(
            client,
            instance_id="inst-1",
            decision_point_id="d1",
            legal_actions=[_OVERCOME, _HOLD_ON],
            timeout_s=5.0,
        )

        self.assertEqual(result, "a-overcome")
        self.assertEqual(len(client.released[0]), 2)

    async def test_transport_failure_propagates_without_followup_cleanup_request(self) -> None:
        client = _FakeClient(
            {"a-overcome": 60.0, "a-hold-on": 53.0},
            transport_fail_action_id="a-hold-on",
        )

        with self.assertRaises(TransportError):
            await best_event_option(
                client,
                instance_id="inst-1",
                decision_point_id="d1",
                legal_actions=[_OVERCOME, _HOLD_ON],
                timeout_s=5.0,
            )

        # Completion is uncertain for the whole batch. The real client retains an exact
        # replay token and blocks any different follow-up request, including cleanup.
        self.assertEqual(client.released, [])


if __name__ == "__main__":
    unittest.main()
