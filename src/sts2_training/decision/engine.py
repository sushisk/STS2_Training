"""`CombatDecisionEngine`: get_decision -> (beam search | heuristic fallback)
-> commit_action, over one `AsyncTrainingApiClient`.

Beam search is only useful for ordinary in-combat decisions. The search engine
checks every explored boundary against `BeamSearchConfig.beam_searchable_action_types`
and returns a no-candidate result when branching is unsupported, at which point
this engine falls back to `HeuristicCombatSelector`.

Expected search failures such as a safely rejected `emulate_actions` batch are
encoded in `BeamSearchResult` and fall back cleanly. Protocol errors, faulted
operations, invalid model outputs, and unexpected model exceptions are surfaced
instead of being converted into heuristic decisions.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sts2_training.api.contract import JsonObject, ROOT_BRANCH_ID
from sts2_training.api.transport import TransportError
from sts2_training.decision.beam_search import BeamSearchConfig, BeamSearchEngine, BeamSearchResult
from sts2_training.decision.policy import PolicyModel, PriorHeuristicPolicy
from sts2_training.decision.value import HeuristicValueFunction, ValueModel
from sts2_training.selection.action_classification import available_actions
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

    This engine is root-only: `commit_action` can only advance the root Branch.
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
        policy_model = policy if policy is not None else PriorHeuristicPolicy()
        value_model = value_fn if value_fn is not None else HeuristicValueFunction()
        self._beam = BeamSearchEngine(
            client, policy=policy_model, value_fn=value_model, config=beam_config
        )
        self._fallback = (
            fallback_selector if fallback_selector is not None else HeuristicCombatSelector()
        )

    @property
    def beam_search(self) -> BeamSearchEngine:
        return self._beam

    async def decide(
        self,
        instance_id: str,
        *,
        timeout_s: float,
    ) -> DecisionOutcome:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        deadline = time.monotonic() + timeout_s

        decision = await self._client.get_decision(
            instance_id, ROOT_BRANCH_ID, timeout_s=timeout_s
        )
        dto = decision.get("masked_emulator_dto")
        if not isinstance(dto, Mapping):
            raise RuntimeError("get_decision returned an invalid masked_emulator_dto")

        raw_legal_actions = dto.get("legal_actions")
        if raw_legal_actions is None:
            legal_actions = []
        elif isinstance(raw_legal_actions, Sequence) and not isinstance(
            raw_legal_actions, (str, bytes)
        ):
            legal_actions = available_actions(raw_legal_actions)
        else:
            raise RuntimeError("get_decision returned invalid legal_actions")

        if not legal_actions:
            return DecisionOutcome(decision, None, "none", None)
        if len(legal_actions) == 1:
            only_action_id = legal_actions[0].get("action_id")
            if not isinstance(only_action_id, str) or not only_action_id:
                raise RuntimeError("available legal action is missing action_id")
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
        timeout_s: float,
    ) -> JsonObject:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        deadline = time.monotonic() + timeout_s
        outcome = await self.decide(
            instance_id,
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
