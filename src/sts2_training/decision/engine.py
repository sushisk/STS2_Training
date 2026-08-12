"""`CombatDecisionEngine`: get_decision -> (event branch search | beam search |
heuristic fallback) -> commit_action, over one `AsyncTrainingApiClient`.

`choice_event_option` decisions are tried first via `event_branch_search.best_event_option`
(disabled with `event_branch_search=False`) - they are never Beam-searchable (see
`combat_decision.COMBAT_BEAM_ACTION_TYPES`), and a static predictive filter alone leaves
HP cost/benefit unweighed beyond a binary lethal/non-lethal cutoff (see that module's
docstring). Falls through to Beam search, then `HeuristicCombatSelector` (whose own
`event_choice_heuristic` hard-avoids confirmed-lethal options) if branch search finds
nothing worth preferring.

Search-budget configuration and Combat decision scope are intentionally separate:
`BeamSearchConfig`/named search modes describe latency-quality tradeoffs, while this
engine owns the default Combat action types eligible for Beam Search. Explicit
`BeamSearchConfig.beam_searchable_action_types` values remain authoritative when a
caller supplies a config, so wrapper construction never silently widens semantic scope.
Structural branch coverage is likewise separate from policy ranking so learned policies
cannot silently change Beam topology.
"""

from __future__ import annotations

import inspect
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from sts2_training.api.contract import JsonObject, ROOT_BRANCH_ID
from sts2_training.api.transport import TransportError
from sts2_training.decision.beam_search import BeamSearchConfig, BeamSearchEngine, BeamSearchResult
from sts2_training.decision.candidate_coverage import CoverageConstrainedPolicy
from sts2_training.decision.combat_decision import COMBAT_BEAM_ACTION_TYPES
from sts2_training.decision.event_branch_search import best_event_option
from sts2_training.decision.policy import PolicyModel, PriorHeuristicPolicy
from sts2_training.decision.search_modes import resolve_search_mode
from sts2_training.decision.value import HeuristicValueFunction, ValueModel
from sts2_training.selection.action_classification import (
    CHOICE_EVENT_OPTION_ACTION_TYPE,
    available_actions,
)
from sts2_training.selection.heuristic_selector import HeuristicCombatSelector, NoAvailableActionError

__all__ = ["CombatDecisionEngine", "DecisionOutcome", "NoAvailableActionError"]


@dataclass(frozen=True)
class DecisionOutcome:
    decision: JsonObject
    chosen_action_id: str | None
    source: str
    beam_result: BeamSearchResult | None


class CombatDecisionEngine:
    """Root-only Combat decision engine.

    With no explicit `beam_config`, the semantic Beam domain defaults to the full Combat
    scope, including interactive continuations. When a caller supplies `beam_config`, its
    `beam_searchable_action_types` is preserved. `beam_action_types` can make the semantic
    scope explicit, but if it is supplied together with a config it must match that
    config; conflicting sources fail fast instead of silently overriding one another.

    Whatever `PolicyModel` is supplied is wrapped in `CoverageConstrainedPolicy`, so
    structural branch-recall invariants survive replacing the heuristic prior with a
    learned or batch-only model. `BeamSearchEngine` owns transport capability and
    Whole Run boundary admission for every frontier depth.
    """

    def __init__(
        self,
        client: Any,
        *,
        policy: PolicyModel | None = None,
        value_fn: ValueModel | None = None,
        beam_config: BeamSearchConfig | None = None,
        beam_action_types: frozenset[str] | None = None,
        fallback_selector: HeuristicCombatSelector | None = None,
        event_branch_search: bool = True,
    ) -> None:
        self._client = client
        self._event_branch_search = event_branch_search
        ranked_policy = policy if policy is not None else PriorHeuristicPolicy()
        policy_model = CoverageConstrainedPolicy(ranked_policy)
        value_model = value_fn if value_fn is not None else HeuristicValueFunction()

        if beam_config is None:
            budget_config = resolve_search_mode()
            resolved_action_types = (
                COMBAT_BEAM_ACTION_TYPES
                if beam_action_types is None
                else frozenset(beam_action_types)
            )
        else:
            budget_config = beam_config
            configured_action_types = frozenset(budget_config.beam_searchable_action_types)
            if beam_action_types is None:
                resolved_action_types = configured_action_types
            else:
                requested_action_types = frozenset(beam_action_types)
                if requested_action_types != configured_action_types:
                    raise ValueError(
                        "beam_action_types conflicts with "
                        "beam_config.beam_searchable_action_types"
                    )
                resolved_action_types = requested_action_types

        resolved_beam_config = replace(
            budget_config,
            beam_searchable_action_types=frozenset(resolved_action_types),
            simulation_options=(
                None
                if budget_config.simulation_options is None
                else dict(budget_config.simulation_options)
            ),
        )
        self._beam = BeamSearchEngine(
            client,
            policy=policy_model,
            value_fn=value_model,
            config=resolved_beam_config,
        )
        self._fallback = (
            fallback_selector if fallback_selector is not None else HeuristicCombatSelector()
        )

    @property
    def client(self) -> Any:
        return self._client

    @property
    def beam_search(self) -> BeamSearchEngine:
        return self._beam

    async def decide(
        self,
        instance_id: str,
        *,
        timeout_s: float,
        decision: Mapping[str, Any] | None = None,
    ) -> DecisionOutcome:
        deadline = _deadline_from_timeout(timeout_s)

        if decision is None:
            raw_decision = await self._client.get_decision(
                instance_id, ROOT_BRANCH_ID, timeout_s=timeout_s
            )
        else:
            raw_decision = decision
        if not isinstance(raw_decision, Mapping):
            raise RuntimeError("get_decision must return a mapping")
        current_decision: JsonObject = dict(raw_decision)

        decision_point_id = current_decision.get("decision_point_id")
        if not isinstance(decision_point_id, str) or not decision_point_id:
            raise RuntimeError("get_decision returned an invalid decision_point_id")

        dto = current_decision.get("masked_emulator_dto")
        if not isinstance(dto, Mapping):
            raise RuntimeError("get_decision returned an invalid masked_emulator_dto")

        raw_legal_actions = dto.get("legal_actions")
        if raw_legal_actions is None:
            if dto.get("run_terminal") is True:
                legal_actions = []
            else:
                raise RuntimeError("get_decision returned invalid legal_actions")
        elif isinstance(raw_legal_actions, Sequence) and not isinstance(
            raw_legal_actions, (str, bytes)
        ):
            validated_actions: list[Mapping[str, Any]] = []
            action_ids: set[str] = set()
            for index, action in enumerate(raw_legal_actions):
                if not isinstance(action, Mapping):
                    raise RuntimeError(f"get_decision returned invalid legal_actions[{index}]")
                action_id = action.get("action_id")
                if not isinstance(action_id, str) or not action_id:
                    raise RuntimeError(
                        f"get_decision returned invalid legal_actions[{index}].action_id"
                    )
                if action_id in action_ids:
                    raise RuntimeError(f"get_decision returned duplicate action_id {action_id!r}")
                action_ids.add(action_id)
                if "is_available" in action and not isinstance(action["is_available"], bool):
                    raise RuntimeError(
                        f"get_decision returned invalid legal_actions[{index}].is_available"
                    )
                validated_actions.append(action)
            legal_actions = available_actions(validated_actions)
        else:
            raise RuntimeError("get_decision returned invalid legal_actions")

        if not legal_actions:
            return DecisionOutcome(current_decision, None, "none", None)
        if len(legal_actions) == 1:
            only_action_id = legal_actions[0].get("action_id")
            if not isinstance(only_action_id, str) or not only_action_id:
                raise RuntimeError("available legal action is missing action_id")
            return DecisionOutcome(current_decision, only_action_id, "forced_single_action", None)

        if self._event_branch_search and any(
            action.get("action_type") == CHOICE_EVENT_OPTION_ACTION_TYPE
            for action in legal_actions
        ):
            remaining = deadline - time.monotonic()
            if remaining > 0:
                best_action_id = await best_event_option(
                    self._client,
                    instance_id=instance_id,
                    decision_point_id=decision_point_id,
                    legal_actions=legal_actions,
                    timeout_s=remaining,
                )
                if best_action_id is not None:
                    return DecisionOutcome(current_decision, best_action_id, "event_branch_search", None)

        result: BeamSearchResult | None = None
        remaining = deadline - time.monotonic()
        if remaining > 0:
            result = await self._beam.search(instance_id, current_decision, timeout_s=remaining)
        if result is not None and _beam_result_is_actionable(result):
            return DecisionOutcome(current_decision, result.best_root_action_id, "beam_search", result)

        chosen = _select_fallback(self._fallback, legal_actions, dto)
        if not isinstance(chosen, Mapping):
            raise RuntimeError("fallback selector must return an available legal action mapping")
        chosen_action_id = chosen.get("action_id")
        available_ids = {action.get("action_id") for action in legal_actions}
        if (
            not isinstance(chosen_action_id, str)
            or not chosen_action_id
            or chosen_action_id not in available_ids
        ):
            raise RuntimeError("fallback selector must return an available legal action")
        return DecisionOutcome(current_decision, chosen_action_id, "heuristic_fallback", result)

    async def decide_and_commit(
        self,
        instance_id: str,
        *,
        timeout_s: float,
        decision: Mapping[str, Any] | None = None,
    ) -> JsonObject:
        deadline = _deadline_from_timeout(timeout_s)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TransportError("decision timeout elapsed before get_decision")
        outcome = await self.decide(instance_id, timeout_s=remaining, decision=decision)
        if outcome.chosen_action_id is None:
            raise NoAvailableActionError(
                "no available legal_actions to select from",
                decision=outcome.decision,
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TransportError("decision timeout elapsed before commit_action")
        return await self._client.commit_action(
            instance_id,
            outcome.decision["decision_point_id"],
            outcome.chosen_action_id,
            timeout_s=remaining,
        )


def _select_fallback(
    fallback_selector: Any,
    legal_actions: Sequence[Mapping[str, Any]],
    masked_emulator_dto: Mapping[str, Any],
) -> Any:
    """Call context-aware selectors without breaking the legacy one-argument contract.

    Signature binding is used before invocation rather than catching ``TypeError`` from
    the selector itself, so a real selector bug is never mistaken for an old signature.
    If a callable does not expose an inspectable signature, the legacy call is the safe
    compatibility default.
    """

    select = fallback_selector.select
    try:
        inspect.signature(select).bind(
            legal_actions,
            masked_emulator_dto=masked_emulator_dto,
        )
    except (TypeError, ValueError):
        return select(legal_actions)
    return select(legal_actions, masked_emulator_dto=masked_emulator_dto)


def _beam_result_is_actionable(result: BeamSearchResult) -> bool:
    if result.best_root_action_id is None:
        return False
    incomplete_depth = result.reason == "time_budget" or result.reason.startswith(
        "emulate_actions_rejected:"
    )
    if not incomplete_depth:
        return True
    return result.best_node is not None and result.best_node.depth <= result.stats.depths_completed


def _deadline_from_timeout(timeout_s: float) -> float:
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or timeout_s <= 0
    ):
        raise ValueError("timeout_s must be a finite positive number")
    return time.monotonic() + float(timeout_s)