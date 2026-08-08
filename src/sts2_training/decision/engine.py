"""`CombatDecisionEngine`: get_decision -> (beam search | heuristic fallback)
-> commit_action, over one `AsyncTrainingApiClient`.

Beam search is only useful for ordinary in-combat decisions. The search engine
checks every explored boundary against `BeamSearchConfig.beam_searchable_action_types`
and returns a no-candidate result when branching is unsupported, at which point
this engine falls back to `HeuristicCombatSelector`.

Expected search failures such as a rejected `emulate_actions` batch are encoded
in `BeamSearchResult` and fall back cleanly. Unexpected exceptions from custom
`PolicyModel`/`ValueModel` implementations are not swallowed: treating a model
bug as a valid heuristic decision would hide correctness problems and make bad
model deployments difficult to detect.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from sts2_training.api.contract import JsonObject, ROOT_BRANCH_ID
from sts2_training.api.transport import TransportError
from sts2_training.decision.beam_search import BeamSearchConfig, BeamSearchEngine, BeamSearchResult
from sts2_training.decision.policy import PolicyModel, PriorHeuristicPolicy
from sts2_training.decision.value import HeuristicValueFunction, ValueModel
from sts2_training.selection.heuristic_selector import HeuristicCombatSelector, NoAvailableActionError

__all__ = ["CombatDecisionEngine", "DecisionOutcome", "NoAvailableActionError"]


@dataclass(frozen=True)
class DecisionOutcome:
    decision: JsonObject
    chosen_action_id: str | None
    source: str  # "beam_search" | "heuristic_fallback" | "forced_single_action" | "none"
    beam_result: BeamSearchResult | None


class CombatDecisionEngine:
    """Construct ONCE per API client/instance and reuse across every real
    decision - `BeamSearchEngine` owns a branch-id/rng-id allocator that must
    stay unique for the instance's whole lifetime (see `BranchIdAllocator`).
    """

    def __init__(
        self,
        client: Any,
        *,
        policy: PolicyModel | None = None,
        value_fn: ValueModel | None = None,
        beam_config: BeamSearchConfig | None = None,
        fallback_selector: HeuristicCombatSelector | None = None,
    ) -> None:
        self._client = client
        self._policy = policy or PriorHeuristicPolicy()
        self._value_fn = value_fn or HeuristicValueFunction()
        self._beam = BeamSearchEngine(
            client, policy=self._policy, value_fn=self._value_fn, config=beam_config
        )
        self._fallback = fallback_selector or HeuristicCombatSelector()

    @property
    def beam_search(self) -> BeamSearchEngine:
        return self._beam

    async def decide(
        self,
        instance_id: str,
        *,
        branch_id: str = ROOT_BRANCH_ID,
        timeout_s: float,
    ) -> DecisionOutcome:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        deadline = time.monotonic() + timeout_s

        decision = await self._client.get_decision(instance_id, branch_id, timeout_s=timeout_s)
        dto = decision.get("masked_emulator_dto") or {}
        legal_actions = dto.get("legal_actions")
        if not legal_actions:
            return DecisionOutcome(decision, None, "none", None)
        if len(legal_actions) == 1:
            only_action_id = legal_actions[0].get("action_id")
            return DecisionOutcome(decision, only_action_id, "forced_single_action", None)

        result: BeamSearchResult | None = None
        remaining = deadline - time.monotonic()
        if remaining > 0:
            result = await self._beam.search(instance_id, decision, timeout_s=remaining)
        if result is not None and result.best_root_action_id is not None:
            return DecisionOutcome(decision, result.best_root_action_id, "beam_search", result)

        chosen = self._fallback.select(legal_actions)
        return DecisionOutcome(decision, chosen["action_id"], "heuristic_fallback", result)

    async def decide_and_commit(
        self,
        instance_id: str,
        *,
        branch_id: str = ROOT_BRANCH_ID,
        timeout_s: float,
    ) -> JsonObject:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        deadline = time.monotonic() + timeout_s
        outcome = await self.decide(
            instance_id,
            branch_id=branch_id,
            timeout_s=deadline - time.monotonic(),
        )
        if outcome.chosen_action_id is None:
            raise NoAvailableActionError("no available legal_actions to select from")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TransportError("decision timeout elapsed before commit_action")
        return await self._client.commit_action(
            instance_id,
            outcome.decision["decision_point_id"],
            outcome.chosen_action_id,
            timeout_s=remaining,
        )
