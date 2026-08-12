"""Coverage for ``best_event_option`` without a live server."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from sts2_training.api.contract import RequestFaultedError, RequestRejectedError
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
        fault_action_id: str | None = None,
        reject_action_id: str | None = None,
        transport_fail_action_id: str | None = None,
        invalidate_on_fault: bool = False,
        invalidate_on_reject: bool = False,
        release_error: Exception | None = None,
        supports_event_batch: bool = True,
        reject_batch: bool = False,
        fault_batch: bool = False,
    ) -> None:
        self.hp_by_action = hp_by_action
        self.fault_action_id = fault_action_id
        self.reject_action_id = reject_action_id
        self.transport_fail_action_id = transport_fail_action_id
        self.invalidate_on_fault = invalidate_on_fault
        self.invalidate_on_reject = invalidate_on_reject
        self.release_error = release_error
        self.reject_batch = reject_batch
        self.fault_batch = fault_batch
        self.emulate_actions_boundaries = (
            frozenset({"event_choice"}) if supports_event_batch else frozenset()
        )
        self.session_invalid = False
        self.pending_retry: object | None = None
        self.emulate_batches: list[list[dict[str, object]]] = []
        self.emulate_singletons: list[dict[str, object]] = []
        self.released: list[list[str]] = []
        self.release_timeouts: list[float] = []
        self.deferred_cleanup: list[tuple[object, str, list[str]]] = []

    def _branch_result(self, item: dict[str, object]) -> dict[str, object]:
        action_id = item["action_id"]
        if action_id == self.fault_action_id:
            return {
                "status": "faulted",
                "error": "candidate simulation failed",
            }
        hp = self.hp_by_action.get(action_id)
        if hp is None:
            return {"status": "faulted"}
        return {
            "status": "completed",
            "masked_emulator_dto": {"hp": hp, "boundary": "map_select"},
        }

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
            self.pending_retry = object()
            raise TransportError("candidate transport failed", completion_uncertain=True)
        if self.reject_batch:
            self.session_invalid = self.invalidate_on_reject
            raise RequestRejectedError(
                {
                    "operation": "emulate_actions",
                    "status": "rejected",
                    "fault_kind": "invalid_action",
                    "error": "batch request rejected",
                }
            )
        if self.fault_batch:
            self.session_invalid = self.invalidate_on_fault
            raise RequestFaultedError(
                {
                    "operation": "emulate_actions",
                    "status": "faulted",
                    "error": "batch request faulted",
                }
            )
        return {
            "status": "completed",
            "branch_results": {
                item["branch_id"]: self._branch_result(item) for item in batch
            },
        }

    async def emulate_action(
        self,
        instance_id,
        parent_branch_id,
        branch_id,
        rng_id,
        decision_point_id,
        action_id,
        *,
        timeout_s,
        simulation_options=None,
    ):
        item = {
            "parent_branch_id": parent_branch_id,
            "branch_id": branch_id,
            "rng_id": rng_id,
            "decision_point_id": decision_point_id,
            "action_id": action_id,
        }
        self.emulate_singletons.append(item)
        if action_id == self.transport_fail_action_id:
            self.pending_retry = object()
            raise TransportError("candidate transport failed", completion_uncertain=True)
        if action_id == self.reject_action_id:
            self.session_invalid = self.invalidate_on_reject
            raise RequestRejectedError(
                {
                    "operation": "emulate_action",
                    "status": "rejected",
                    "fault_kind": "invalid_action",
                    "error": "candidate request rejected",
                }
            )
        if action_id == self.fault_action_id:
            self.session_invalid = self.invalidate_on_fault
            raise RequestFaultedError(
                {
                    "operation": "emulate_action",
                    "status": "faulted",
                    "error": "candidate simulation failed",
                }
            )
        return self._branch_result(item)

    def defer_branch_cleanup_after_retry(self, retry, instance_id, branch_ids):
        self.deferred_cleanup.append((retry, instance_id, list(branch_ids)))

    async def release_branches(self, instance_id, branch_ids, *, timeout_s):
        self.released.append(list(branch_ids))
        self.release_timeouts.append(timeout_s)
        if self.release_error is not None:
            if (
                isinstance(self.release_error, TransportError)
                and self.release_error.completion_uncertain
            ):
                self.pending_retry = object()
            raise self.release_error
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
        self.assertEqual(client.emulate_singletons, [])

    async def test_one_branch_fault_in_batch_does_not_sink_the_others(self) -> None:
        client = _FakeClient(
            {"a-overcome": 60.0, "a-hold-on": 53.0},
            fault_action_id="a-hold-on",
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

    async def test_pre_map_batch_rejection_returns_none_without_cleanup(self) -> None:
        client = _FakeClient({}, reject_batch=True)

        result = await best_event_option(
            client,
            instance_id="inst-1",
            decision_point_id="d1",
            legal_actions=[_OVERCOME, _HOLD_ON],
            timeout_s=5.0,
        )

        self.assertIsNone(result)
        self.assertEqual(len(client.emulate_batches), 1)
        self.assertEqual(client.released, [])
        self.assertEqual(client.deferred_cleanup, [])

    async def test_session_invalid_batch_rejection_propagates(self) -> None:
        client = _FakeClient(
            {},
            reject_batch=True,
            invalidate_on_reject=True,
        )

        with self.assertRaises(RequestRejectedError):
            await best_event_option(
                client,
                instance_id="inst-1",
                decision_point_id="d1",
                legal_actions=[_OVERCOME, _HOLD_ON],
                timeout_s=5.0,
            )

        self.assertEqual(client.released, [])

    async def test_request_level_batch_fault_is_cleaned_up_and_returns_none(self) -> None:
        client = _FakeClient({}, fault_batch=True)

        result = await best_event_option(
            client,
            instance_id="inst-1",
            decision_point_id="d1",
            legal_actions=[_OVERCOME, _HOLD_ON],
            timeout_s=5.0,
        )

        self.assertIsNone(result)
        batch_ids = [item["branch_id"] for item in client.emulate_batches[0]]
        self.assertEqual(client.released, [batch_ids])

    async def test_missing_event_batch_capability_falls_back_to_singletons(self) -> None:
        client = _FakeClient(
            {"a-overcome": 60.0, "a-hold-on": 53.0},
            supports_event_batch=False,
        )

        result = await best_event_option(
            client,
            instance_id="inst-1",
            decision_point_id="d1",
            legal_actions=[_OVERCOME, _HOLD_ON],
            timeout_s=5.0,
        )

        self.assertEqual(result, "a-overcome")
        self.assertEqual(client.emulate_batches, [])
        self.assertEqual(len(client.emulate_singletons), 2)
        self.assertEqual(
            [item["rng_id"] for item in client.emulate_singletons],
            [1, 2],
        )
        self.assertEqual(len(client.released[0]), 2)

    async def test_singleton_rejection_does_not_sink_the_other_candidate(self) -> None:
        client = _FakeClient(
            {"a-overcome": 60.0, "a-hold-on": 53.0},
            reject_action_id="a-hold-on",
            supports_event_batch=False,
        )

        result = await best_event_option(
            client,
            instance_id="inst-1",
            decision_point_id="d1",
            legal_actions=[_OVERCOME, _HOLD_ON],
            timeout_s=5.0,
        )

        self.assertEqual(result, "a-overcome")
        self.assertEqual(len(client.emulate_singletons), 2)
        self.assertEqual(len(client.released), 1)
        self.assertEqual(len(client.released[0]), 1)
        released_id = client.released[0][0]
        successful_id = client.emulate_singletons[0]["branch_id"]
        self.assertEqual(released_id, successful_id)

    async def test_all_singleton_candidates_rejected_returns_none(self) -> None:
        client = _FakeClient({}, supports_event_batch=False)

        async def reject_both(*args, **kwargs):
            raise RequestRejectedError(
                {
                    "operation": "emulate_action",
                    "status": "rejected",
                    "fault_kind": "invalid_action",
                    "error": "parent_branch_id has not reached a map_select boundary yet",
                }
            )

        client.emulate_action = reject_both  # type: ignore[method-assign]

        result = await best_event_option(
            client,
            instance_id="inst-1",
            decision_point_id="d1",
            legal_actions=[_OVERCOME, _HOLD_ON],
            timeout_s=5.0,
        )

        self.assertIsNone(result)
        self.assertEqual(client.released, [])

    async def test_session_invalid_singleton_rejection_propagates(self) -> None:
        client = _FakeClient(
            {"a-overcome": 60.0, "a-hold-on": 53.0},
            reject_action_id="a-hold-on",
            invalidate_on_reject=True,
            supports_event_batch=False,
        )

        with self.assertRaises(RequestRejectedError):
            await best_event_option(
                client,
                instance_id="inst-1",
                decision_point_id="d1",
                legal_actions=[_OVERCOME, _HOLD_ON],
                timeout_s=5.0,
            )

        self.assertEqual(client.released, [])

    async def test_session_invalid_singleton_fault_propagates(self) -> None:
        client = _FakeClient(
            {"a-overcome": 60.0, "a-hold-on": 53.0},
            fault_action_id="a-hold-on",
            invalidate_on_fault=True,
            supports_event_batch=False,
        )

        with self.assertRaises(RequestFaultedError):
            await best_event_option(
                client,
                instance_id="inst-1",
                decision_point_id="d1",
                legal_actions=[_OVERCOME, _HOLD_ON],
                timeout_s=5.0,
            )

        self.assertEqual(client.released, [])

    async def test_transport_uncertainty_defers_full_batch_cleanup_to_exact_retry(self) -> None:
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

        self.assertEqual(client.released, [])
        self.assertEqual(len(client.deferred_cleanup), 1)
        _, instance_id, cleanup_ids = client.deferred_cleanup[0]
        self.assertEqual(instance_id, "inst-1")
        batch_ids = [item["branch_id"] for item in client.emulate_batches[0]]
        self.assertEqual(cleanup_ids, batch_ids)

    async def test_singleton_transport_uncertainty_carries_prior_owned_branch(self) -> None:
        client = _FakeClient(
            {"a-overcome": 60.0, "a-hold-on": 53.0},
            transport_fail_action_id="a-hold-on",
            supports_event_batch=False,
        )

        with self.assertRaises(TransportError):
            await best_event_option(
                client,
                instance_id="inst-1",
                decision_point_id="d1",
                legal_actions=[_OVERCOME, _HOLD_ON],
                timeout_s=5.0,
            )

        self.assertEqual(client.released, [])
        self.assertEqual(len(client.deferred_cleanup), 1)
        _, _, cleanup_ids = client.deferred_cleanup[0]
        singleton_ids = [item["branch_id"] for item in client.emulate_singletons]
        self.assertEqual(cleanup_ids, singleton_ids)

    async def test_cleanup_completion_uncertain_transport_failure_propagates(self) -> None:
        client = _FakeClient(
            {"a-overcome": 60.0, "a-hold-on": 53.0},
            release_error=TransportError("cleanup failed", completion_uncertain=True),
        )

        with self.assertRaises(TransportError):
            await best_event_option(
                client,
                instance_id="inst-1",
                decision_point_id="d1",
                legal_actions=[_OVERCOME, _HOLD_ON],
                timeout_s=5.0,
            )

        self.assertIsNotNone(client.pending_retry)

    async def test_reserves_deadline_for_cleanup_when_singleton_budget_expires(self) -> None:
        clock = [0.0]
        client = _FakeClient(
            {"a-overcome": 60.0, "a-hold-on": 53.0},
            supports_event_batch=False,
        )
        original_emulate = client.emulate_action

        async def consume_evaluation_budget(*args, **kwargs):
            result = await original_emulate(*args, **kwargs)
            clock[0] += kwargs["timeout_s"]
            return result

        client.emulate_action = consume_evaluation_budget  # type: ignore[method-assign]

        with patch(
            "sts2_training.decision.event_branch_search.time.monotonic",
            side_effect=lambda: clock[0],
        ):
            result = await best_event_option(
                client,
                instance_id="inst-1",
                decision_point_id="d1",
                legal_actions=[_OVERCOME, _HOLD_ON],
                timeout_s=5.0,
            )

        self.assertIsNone(result)
        self.assertEqual(len(client.emulate_singletons), 1)
        self.assertEqual(len(client.released), 1)
        self.assertEqual(len(client.released[0]), 1)
        self.assertGreater(client.release_timeouts[0], 0.0)

    async def test_rejects_invalid_timeout(self) -> None:
        client = _FakeClient({})

        for timeout_s in (0.0, -1.0, float("inf"), float("nan"), True):
            with self.subTest(timeout_s=timeout_s):
                with self.assertRaises(ValueError):
                    await best_event_option(
                        client,
                        instance_id="i",
                        decision_point_id="d",
                        legal_actions=[_OVERCOME, _HOLD_ON],
                        timeout_s=timeout_s,
                    )


if __name__ == "__main__":
    unittest.main()
