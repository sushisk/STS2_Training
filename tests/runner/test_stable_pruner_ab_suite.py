from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.decision.stable_pruner import StableFrontierPruner
from sts2_training.runner.stable_pruner_ab import (
    StablePrunerABArmResult,
    StablePrunerABPairResult,
    StablePrunerABReport,
    summarize_pairs,
)
from sts2_training.runner.stable_pruner_ab_suite import (
    StablePrunerABSuiteRunner,
    load_suite_manifest,
)


class _LearnedPruner(StableFrontierPruner):
    name = "learned"
    version = "v1"

    def select(self, frontier, *, k, context):
        del context
        return list(frontier[:k])


class _FakeABRunner:
    def __init__(self, client, *, scenario, learned_pruner, beam_config, **kwargs):
        del client, learned_pruner, beam_config, kwargs
        self._character_id = scenario.character_id

    async def run(self, seeds):
        pairs = []
        for seed in seeds:
            learned_wins = self._character_id == "LEARNED_CASE"
            baseline = _arm("baseline", seed, "defeat" if learned_wins else "victory")
            learned = _arm("learned", seed, "victory" if learned_wins else "defeat")
            pairs.append(
                StablePrunerABPairResult(
                    seed=seed,
                    baseline=baseline,
                    learned=learned,
                    common_action_prefix=0,
                    first_divergence_index=0,
                    winner="learned" if learned_wins else "baseline",
                )
            )
        pair_tuple = tuple(pairs)
        return StablePrunerABReport(
            schema_version=1,
            scenario_template_sha256=f"scenario-{self._character_id}",
            seeds=tuple(seeds),
            search_config={"beam_width": 4},
            arm_order="alternate",
            learned_pruner_name="learned",
            learned_pruner_version="v1",
            learned_artifact_sha256=None,
            pairs=pair_tuple,
            summary=summarize_pairs(pair_tuple),
        )


def _arm(arm: str, seed: int, outcome: str) -> StablePrunerABArmResult:
    return StablePrunerABArmResult(
        arm=arm,
        seed=seed,
        pruner_name=arm,
        pruner_version="1",
        outcome=outcome,
        decisions_made=1,
        elapsed_s=0.1,
        beam_decisions=1,
        nodes_expanded=2,
        branches_created=2,
        beam_total_ms=3.0,
        heuristic_fallbacks=0,
        decisions=(),
    )


def _scenario_payload(character_id: str) -> dict:
    return {
        "character_id": character_id,
        "player_hp": 40,
        "player_max_hp": 40,
        "hand": [],
        "draw_pile": [],
        "discard_pile": [],
        "enemies": [{"monster_id": "TEST_ENEMY", "hp": 10}],
        "seed": 1,
    }


class StablePrunerABSuiteManifestTest(unittest.TestCase):
    def test_resolves_relative_scenarios_and_returns_canonical_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scenarios").mkdir()
            manifest = root / "suite.json"
            manifest.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "name": "case-a",
                                "scenario": "scenarios/a.json",
                                "seeds": [1, 2],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            cases, digest = load_suite_manifest(manifest)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].name, "case-a")
        self.assertEqual(cases[0].seeds, (1, 2))
        self.assertTrue(cases[0].scenario_path.is_absolute())
        self.assertEqual(len(digest), 64)

    def test_duplicate_case_name_and_seed_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            duplicate_name = root / "duplicate-name.json"
            duplicate_name.write_text(
                json.dumps(
                    {
                        "cases": [
                            {"name": "same", "scenario": "a.json", "seeds": [1]},
                            {"name": "same", "scenario": "b.json", "seeds": [2]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate.*case name"):
                load_suite_manifest(duplicate_name)

            duplicate_seed = root / "duplicate-seed.json"
            duplicate_seed.write_text(
                json.dumps(
                    {
                        "cases": [
                            {"name": "a", "scenario": "a.json", "seeds": [3, 3]}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate seed"):
                load_suite_manifest(duplicate_seed)


class StablePrunerABSuiteRunnerTest(unittest.IsolatedAsyncioTestCase):
    async def test_aggregates_pairs_across_scenarios_without_losing_case_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            learned_path = root / "learned.json"
            baseline_path = root / "baseline.json"
            learned_path.write_text(
                json.dumps(_scenario_payload("LEARNED_CASE")), encoding="utf-8"
            )
            baseline_path.write_text(
                json.dumps(_scenario_payload("BASELINE_CASE")), encoding="utf-8"
            )
            manifest_path = root / "suite.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "name": "learned-case",
                                "scenario": "learned.json",
                                "seeds": [10, 11],
                            },
                            {
                                "name": "baseline-case",
                                "scenario": "baseline.json",
                                "seeds": [20],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            cases, digest = load_suite_manifest(manifest_path)

            with patch(
                "sts2_training.runner.stable_pruner_ab_suite.StablePrunerABRunner",
                _FakeABRunner,
            ):
                report = await StablePrunerABSuiteRunner(
                    object(),
                    learned_pruner=_LearnedPruner(),
                    beam_config=BeamSearchConfig(beam_width=4, top_k_actions=2, max_depth=2),
                ).run(cases, manifest_sha256=digest)

        self.assertEqual(report.manifest_sha256, digest)
        self.assertEqual(len(report.cases), 2)
        self.assertEqual(report.cases[0].name, "learned-case")
        self.assertEqual(report.cases[0].report.summary.learned_wins, 2)
        self.assertEqual(report.cases[1].report.summary.baseline_wins, 1)
        self.assertEqual(report.aggregate.pairs, 3)
        self.assertEqual(report.aggregate.learned_wins, 2)
        self.assertEqual(report.aggregate.baseline_wins, 1)
        self.assertEqual(report.aggregate.diverged_pairs, 3)
        self.assertEqual(report.outcome_statistics.discordant_pairs, 3)
        self.assertAlmostEqual(report.outcome_statistics.learned_share_of_discordant, 2 / 3)
        self.assertEqual(
            report.outcome_statistics.two_sided_exact_sign_test_p_value,
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
