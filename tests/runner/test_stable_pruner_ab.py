from __future__ import annotations

import json
import unittest

from sts2_training.decision.beam_search import BeamSearchConfig, BeamSearchResult, BeamSearchStats
from sts2_training.decision.engine import DecisionOutcome
from sts2_training.decision.stable_pruner import StableFrontierPruner
from sts2_training.runner.scenario import CombatScenario, EnemyScenario
from sts2_training.runner.stable_pruner_ab import (
    StablePrunerABArmResult,
    StablePrunerABDecision,
    StablePrunerABRunner,
    _parse_seed_list,
    compare_arm_results,
    summarize_pairs,
)


class _LearnedTestPruner(StableFrontierPruner):
    name = "learned_test"
    version = "test-v1"
    artifact_metadata = {"artifact_sha256": "artifact-test-sha"}

    def select(self, frontier, *, k, context):
        del context
        return list(frontier[:k])


class _FakeClient:
    def __init__(self) -> None:
        self._counter = 0
        self.seeds: list[int] = []
        self.commits: list[str] = []
        self.closed: list[str] = []

    async def start_instance(self, config, *, timeout_s):
        del timeout_s
        self._counter += 1
        instance_id = f"inst-{self._counter}"
        self.seeds.append(config["seed"])
        return instance_id

    async def get_decision(self, instance_id, branch_id, *, timeout_s):
        del branch_id, timeout_s
        return {
            "decision_point_id": f"{instance_id}-root",
            "masked_emulator_dto": {
                "legal_actions": [
                    {
                        "action_id": "baseline_action",
                        "action_type": "card",
                        "card_id": "STRIKE",
                        "is_available": True,
                    },
                    {
                        "action_id": "learned_action",
                        "action_type": "card",
                        "card_id": "BASH",
                        "is_available": True,
                    },
                ]
            },
        }

    async def commit_action(self, instance_id, decision_point_id, action_id, *, timeout_s):
        del decision_point_id, timeout_s
        self.commits.append(action_id)
        outcome = "victory" if action_id == "learned_action" else "defeat"
        return {
            "decision_point_id": f"{instance_id}-terminal",
            "masked_emulator_dto": {
                "terminal": True,
                "outcome": outcome,
                "legal_actions": [],
            },
        }

    async def close_instance(self, instance_id, *, timeout_s):
        del timeout_s
        self.closed.append(instance_id)


class _FakeEngine:
    def __init__(self, pruner) -> None:
        self._pruner = pruner

    async def decide(self, instance_id, *, timeout_s, decision):
        del instance_id, timeout_s
        learned = self._pruner.name == "learned_test"
        action_id = "learned_action" if learned else "baseline_action"
        stats = BeamSearchStats(
            depths_completed=2,
            nodes_expanded=2 if learned else 3,
            branches_created=4 if learned else 5,
            total_ms=7.0 if learned else 9.0,
        )
        result = BeamSearchResult(
            best_root_action_id=action_id,
            best_value=2.0 if learned else 1.0,
            best_node=None,
            reason="max_depth",
            stats=stats,
        )
        return DecisionOutcome(dict(decision), action_id, "beam_search", result)


class StablePrunerABRunnerTest(unittest.IsolatedAsyncioTestCase):
    async def test_runs_same_seed_for_both_arms_and_alternates_execution_order(self) -> None:
        client = _FakeClient()
        configs: list[BeamSearchConfig] = []

        def engine_factory(_client, pruner, config):
            configs.append(config)
            return _FakeEngine(pruner)

        scenario = CombatScenario(
            character_id="test-character",
            player_hp=40,
            player_max_hp=40,
            hand=(),
            draw_pile=(),
            discard_pile=(),
            enemies=(EnemyScenario(monster_id="test-enemy", hp=10),),
            seed=1,
        )
        config = BeamSearchConfig(
            beam_width=4,
            top_k_actions=2,
            max_depth=2,
            simulation_options={"stop_condition": "next_decision"},
        )
        report = await StablePrunerABRunner(
            client,
            scenario=scenario,
            learned_pruner=_LearnedTestPruner(),
            beam_config=config,
            arm_order="alternate",
            engine_factory=engine_factory,
        ).run((7, 8))

        self.assertEqual(client.seeds, [7, 7, 8, 8])
        self.assertEqual(
            client.commits,
            ["baseline_action", "learned_action", "learned_action", "baseline_action"],
        )
        self.assertEqual(len(client.closed), 4)
        self.assertEqual(report.seeds, (7, 8))
        self.assertEqual(len(report.scenario_template_sha256), 64)
        self.assertEqual(report.learned_pruner_name, "learned_test")
        self.assertEqual(report.learned_pruner_version, "test-v1")
        self.assertEqual(report.learned_artifact_sha256, "artifact-test-sha")
        self.assertEqual(report.summary.pairs, 2)
        self.assertEqual(report.summary.learned_wins, 2)
        self.assertEqual(report.summary.baseline_wins, 0)
        self.assertEqual(report.summary.diverged_pairs, 2)
        self.assertEqual(report.summary.baseline_mean_nodes_expanded, 3.0)
        self.assertEqual(report.summary.learned_mean_nodes_expanded, 2.0)
        self.assertEqual(report.pairs[0].baseline.pruner_name, "value_top_k")
        self.assertEqual(report.pairs[0].learned.pruner_name, "learned_test")
        self.assertEqual(report.pairs[0].baseline.decisions[0].action_semantics["card_id"], "STRIKE")
        self.assertEqual(report.pairs[0].learned.decisions[0].action_semantics["card_id"], "BASH")
        self.assertEqual(len(configs), 4)
        for arm_config in configs:
            self.assertIsNot(arm_config, config)
            self.assertEqual(arm_config.beam_width, config.beam_width)
            self.assertEqual(arm_config.top_k_actions, config.top_k_actions)
            self.assertEqual(arm_config.max_depth, config.max_depth)
            self.assertEqual(arm_config.simulation_options, config.simulation_options)


class StablePrunerABComparisonTest(unittest.TestCase):
    def test_action_comparison_stops_at_first_semantic_divergence(self) -> None:
        baseline = _arm("baseline", "victory", ("same", "baseline-only"))
        learned = _arm("learned", "victory", ("same", "learned-only"))

        pair = compare_arm_results(seed=9, baseline=baseline, learned=learned)

        self.assertEqual(pair.common_action_prefix, 1)
        self.assertEqual(pair.first_divergence_index, 1)
        self.assertEqual(pair.winner, "tie")

    def test_decision_local_action_ids_do_not_create_false_divergence(self) -> None:
        baseline = _arm("baseline", "victory", ("same",))
        learned = _arm("learned", "victory", ("same",))

        self.assertNotEqual(
            baseline.decisions[0].chosen_action_id,
            learned.decisions[0].chosen_action_id,
        )
        pair = compare_arm_results(seed=11, baseline=baseline, learned=learned)
        self.assertEqual(pair.common_action_prefix, 1)
        self.assertIsNone(pair.first_divergence_index)

    def test_unknown_terminal_outcome_is_not_scored_as_a_win_or_loss(self) -> None:
        pair = compare_arm_results(
            seed=3,
            baseline=_arm("baseline", "aborted", ("a",)),
            learned=_arm("learned", "victory", ("a",)),
        )

        self.assertIsNone(pair.winner)
        summary = summarize_pairs((pair,))
        self.assertEqual(summary.resolved_outcome_pairs, 0)
        self.assertEqual(summary.unknown_outcome_pairs, 1)

    def test_seed_parser_accepts_signed_integer_list(self) -> None:
        self.assertEqual(_parse_seed_list("1, 2,-3"), (1, 2, -3))


def _arm(arm: str, outcome: str | None, actions: tuple[str, ...]) -> StablePrunerABArmResult:
    decisions = []
    for index, action in enumerate(actions):
        semantics = {"action_type": "card", "card_id": action}
        decisions.append(
            StablePrunerABDecision(
                index=index,
                decision_point_id=f"d-{arm}-{index}",
                chosen_action_id=f"opaque-{arm}-{index}",
                chosen_action_type="card",
                action_semantics=semantics,
                action_signature=json.dumps(
                    semantics,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                source="beam_search",
                beam_reason="max_depth",
                beam_best_value=1.0,
                nodes_expanded=1,
                branches_created=1,
                beam_total_ms=2.0,
            )
        )
    decision_tuple = tuple(decisions)
    return StablePrunerABArmResult(
        arm=arm,
        seed=1,
        pruner_name=arm,
        pruner_version="1",
        outcome=outcome,
        decisions_made=len(decision_tuple),
        elapsed_s=0.5,
        beam_decisions=len(decision_tuple),
        nodes_expanded=len(decision_tuple),
        branches_created=len(decision_tuple),
        beam_total_ms=2.0 * len(decision_tuple),
        heuristic_fallbacks=0,
        decisions=decision_tuple,
    )


if __name__ == "__main__":
    unittest.main()
