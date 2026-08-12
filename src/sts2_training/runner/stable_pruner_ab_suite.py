"""Run fixed-seed stable-pruner A/B evaluation across multiple Combat scenarios.

Manifest format::

    {
      "cases": [
        {"name": "slime", "scenario": "scenarios/slime.json", "seeds": [101, 102]},
        {"name": "cultist", "scenario": "scenarios/cultist.json", "seeds": [201, 202]}
      ]
    }

Scenario paths are resolved relative to the manifest. Every case uses the same learned
artifact, BeamSearchConfig, arm-order policy, and A/B semantics from ``stable_pruner_ab``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from sts2_training.api import AsyncTrainingApiClient, TcpConnection
from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.decision.combat_decision import COMBAT_BEAM_ACTION_TYPES
from sts2_training.decision.learned_pruner import LinearStableFrontierPruner
from sts2_training.decision.search_modes import resolve_search_mode
from sts2_training.decision.stable_pruner import StableFrontierPruner
from sts2_training.runner._cli import add_common_arguments, configure_logging
from sts2_training.runner.scenario import CombatScenario, EnemyScenario
from sts2_training.runner.stable_pruner_ab import (
    EngineFactory,
    StablePrunerABReport,
    StablePrunerABRunner,
    StablePrunerABSummary,
    summarize_pairs,
)
from sts2_training.runner.stable_pruner_ab_stats import (
    PairedOutcomeStatistics,
    paired_outcome_statistics,
)

AB_SUITE_REPORT_SCHEMA_VERSION = 1
_ARM_ORDERS = ("alternate", "baseline-first", "learned-first")


@dataclass(frozen=True)
class StablePrunerABCaseSpec:
    name: str
    scenario_path: Path
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class StablePrunerABCaseResult:
    name: str
    scenario_path: str
    report: StablePrunerABReport


@dataclass(frozen=True)
class StablePrunerABSuiteReport:
    schema_version: int
    manifest_sha256: str
    cases: tuple[StablePrunerABCaseResult, ...]
    aggregate: StablePrunerABSummary
    outcome_statistics: PairedOutcomeStatistics


class StablePrunerABSuiteRunner:
    """Evaluate one learned pruner over a manifest-defined Combat scenario suite."""

    def __init__(
        self,
        client: Any,
        *,
        learned_pruner: StableFrontierPruner,
        beam_config: BeamSearchConfig,
        start_timeout_s: float = 30.0,
        decision_timeout_s: float = 30.0,
        max_decisions: int | None = None,
        arm_order: str = "alternate",
        engine_factory: EngineFactory | None = None,
    ) -> None:
        if arm_order not in _ARM_ORDERS:
            raise ValueError(f"arm_order must be one of {_ARM_ORDERS}")
        self._client = client
        self._learned_pruner = learned_pruner
        self._beam_config = _clone_beam_config(beam_config)
        self._start_timeout_s = start_timeout_s
        self._decision_timeout_s = decision_timeout_s
        self._max_decisions = max_decisions
        self._arm_order = arm_order
        self._engine_factory = engine_factory

    async def run(
        self,
        cases: Sequence[StablePrunerABCaseSpec],
        *,
        manifest_sha256: str,
    ) -> StablePrunerABSuiteReport:
        if not cases:
            raise ValueError("A/B suite requires at least one case")
        if not isinstance(manifest_sha256, str) or not manifest_sha256:
            raise ValueError("manifest_sha256 must be non-empty")

        case_results: list[StablePrunerABCaseResult] = []
        all_pairs = []
        for case in cases:
            scenario = load_combat_scenario(case.scenario_path)
            report = await StablePrunerABRunner(
                self._client,
                scenario=scenario,
                learned_pruner=self._learned_pruner,
                beam_config=_clone_beam_config(self._beam_config),
                start_timeout_s=self._start_timeout_s,
                decision_timeout_s=self._decision_timeout_s,
                max_decisions=self._max_decisions,
                arm_order=self._arm_order,
                engine_factory=self._engine_factory,
            ).run(case.seeds)
            case_results.append(
                StablePrunerABCaseResult(
                    name=case.name,
                    scenario_path=str(case.scenario_path),
                    report=report,
                )
            )
            all_pairs.extend(report.pairs)

        pair_tuple = tuple(all_pairs)
        return StablePrunerABSuiteReport(
            schema_version=AB_SUITE_REPORT_SCHEMA_VERSION,
            manifest_sha256=manifest_sha256,
            cases=tuple(case_results),
            aggregate=summarize_pairs(pair_tuple),
            outcome_statistics=paired_outcome_statistics(pair_tuple),
        )


def load_suite_manifest(path: str | Path) -> tuple[tuple[StablePrunerABCaseSpec, ...], str]:
    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("A/B suite manifest root must be an object")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes, bytearray)):
        raise ValueError("A/B suite manifest must contain a cases sequence")
    if not raw_cases:
        raise ValueError("A/B suite manifest cases must not be empty")

    cases: list[StablePrunerABCaseSpec] = []
    seen_names: set[str] = set()
    canonical_cases: list[dict[str, Any]] = []
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, Mapping):
            raise ValueError(f"A/B suite cases[{index}] must be an object")
        name = raw_case.get("name")
        scenario_value = raw_case.get("scenario")
        seeds_value = raw_case.get("seeds")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"A/B suite cases[{index}].name must be non-empty")
        name = name.strip()
        if name in seen_names:
            raise ValueError(f"duplicate A/B suite case name: {name!r}")
        seen_names.add(name)
        if not isinstance(scenario_value, str) or not scenario_value.strip():
            raise ValueError(f"A/B suite cases[{index}].scenario must be non-empty")
        seeds = _validate_manifest_seeds(seeds_value, index=index)
        scenario_relative = scenario_value.strip()
        scenario_path = (manifest_path.parent / scenario_relative).resolve()
        cases.append(
            StablePrunerABCaseSpec(
                name=name,
                scenario_path=scenario_path,
                seeds=seeds,
            )
        )
        canonical_cases.append(
            {
                "name": name,
                "scenario": scenario_relative,
                "seeds": list(seeds),
            }
        )

    canonical = json.dumps(
        {"cases": canonical_cases},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return tuple(cases), hashlib.sha256(canonical).hexdigest()


def load_combat_scenario(path: str | Path) -> CombatScenario:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Combat scenario JSON root must be an object")
    fields = dict(payload)
    raw_enemies = fields.get("enemies")
    if not isinstance(raw_enemies, Sequence) or isinstance(
        raw_enemies, (str, bytes, bytearray)
    ):
        raise ValueError("Combat scenario JSON must contain enemies as a sequence")
    normalized_enemies = []
    for index, enemy in enumerate(raw_enemies):
        if not isinstance(enemy, Mapping):
            raise ValueError(f"Combat scenario enemies[{index}] must be an object")
        normalized_enemies.append(EnemyScenario(**dict(enemy)))
    fields["enemies"] = normalized_enemies
    return CombatScenario(**fields)


def suite_report_to_json(report: StablePrunerABSuiteReport) -> str:
    return json.dumps(asdict(report), ensure_ascii=False, sort_keys=True)


def _validate_manifest_seeds(value: Any, *, index: int) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"A/B suite cases[{index}].seeds must be a sequence")
    if not value:
        raise ValueError(f"A/B suite cases[{index}].seeds must not be empty")
    seeds: list[int] = []
    seen: set[int] = set()
    for seed_index, seed in enumerate(value):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(
                f"A/B suite cases[{index}].seeds[{seed_index}] must be an integer"
            )
        if seed in seen:
            raise ValueError(f"duplicate seed {seed} in A/B suite case index {index}")
        seen.add(seed)
        seeds.append(seed)
    return tuple(seeds)


def _clone_beam_config(config: BeamSearchConfig) -> BeamSearchConfig:
    return replace(
        config,
        beam_searchable_action_types=frozenset(config.beam_searchable_action_types),
        simulation_options=(
            None if config.simulation_options is None else dict(config.simulation_options)
        ),
    )


def _cli_beam_config(args: argparse.Namespace) -> BeamSearchConfig:
    config = resolve_search_mode(args.search_mode, max_depth=args.beam_depth)
    return replace(
        config,
        beam_searchable_action_types=COMBAT_BEAM_ACTION_TYPES,
        simulation_options=(
            None if config.simulation_options is None else dict(config.simulation_options)
        ),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--arm-order",
        choices=_ARM_ORDERS,
        default="alternate",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> StablePrunerABSuiteReport:
    cases, manifest_sha256 = load_suite_manifest(args.manifest)
    learned = LinearStableFrontierPruner.from_weights_file(args.weights)
    connection = TcpConnection(
        host=args.host,
        port=args.port,
        connect_timeout_s=args.connect_timeout,
    )
    async with AsyncTrainingApiClient(connection) as client:
        return await StablePrunerABSuiteRunner(
            client,
            learned_pruner=learned,
            beam_config=_cli_beam_config(args),
            decision_timeout_s=args.decision_timeout,
            max_decisions=args.max_decisions,
            arm_order=args.arm_order,
        ).run(cases, manifest_sha256=manifest_sha256)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging(args.log_level)
    report = asyncio.run(_run(args))
    encoded = suite_report_to_json(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
