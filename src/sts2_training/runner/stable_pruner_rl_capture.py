"""Attach frozen terminal resource snapshots to StablePruner RL A/B pairs."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from sts2_training.runner.combat_resource_reward import combat_resource_snapshot
from sts2_training.runner.stable_pruner_ab import StablePrunerABRunner
from sts2_training.runner.terminal_dto_capture import TerminalDtoCapture


class ResourceCapturingABRunner(StablePrunerABRunner):
    def __init__(
        self,
        client: Any,
        *,
        scenario: Any,
        arm_order: str = "alternate",
        **kwargs: Any,
    ) -> None:
        self._capture = TerminalDtoCapture(client)
        self._reward_scenario = scenario
        self._reward_arm_order = arm_order
        super().__init__(self._capture, scenario=scenario, arm_order=arm_order, **kwargs)

    async def run(self, seeds: Any) -> Any:
        self._capture.terminal_dtos.clear()
        self._capture.final_dto = None
        report = await super().run(seeds)
        if len(report.pairs) != 1 or len(self._capture.terminal_dtos) != 2:
            raise ValueError("RL collection requires exactly one paired terminal result")
        first, second = self._capture.terminal_dtos
        if self._reward_arm_order == "learned-first":
            baseline, learned = second, first
        else:
            baseline, learned = first, second
        initial = sum(item is not None for item in self._reward_scenario.potions)
        pair = enrich_pair(
            report.pairs[0],
            baseline_dto=baseline,
            learned_dto=learned,
            initial_potion_count=initial,
        )
        return SimpleNamespace(**report.__dict__, pairs=(pair,))


def enrich_pair(
    pair: Any,
    *,
    baseline_dto: dict[str, Any],
    learned_dto: dict[str, Any],
    initial_potion_count: int,
) -> Any:
    payload = dict(pair.__dict__)
    payload["baseline"] = _enrich_arm(pair.baseline, baseline_dto, initial_potion_count)
    payload["learned"] = _enrich_arm(pair.learned, learned_dto, initial_potion_count)
    return SimpleNamespace(**payload)


def _enrich_arm(arm: Any, dto: dict[str, Any], initial: int) -> Any:
    snapshot = combat_resource_snapshot(dto, initial_potion_count=initial)
    return SimpleNamespace(
        **arm.__dict__,
        terminal_hp=snapshot.hp,
        terminal_max_hp=snapshot.max_hp,
        terminal_potion_count=snapshot.potion_count,
        initial_potion_count=snapshot.initial_potion_count,
    )
