"""Collect budgeted Oracle targets at each root Combat decision.

The Oracle only observes via branch emulation; the committed root action is still chosen
by a normal ``CombatDecisionEngine``. This keeps the collected state distribution tied to
the runtime policy/search configuration instead of silently switching play to the wider
teacher search.

CLI example::

    python -m sts2_training.runner.oracle_collection \\
        --scenario combat.json --output data/combat_oracle.jsonl \\
        --oracle-beam-width 32 --oracle-depth 4 --target-beam-width 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from sts2_training.api import AsyncTrainingApiClient, TcpConnection
from sts2_training.api.contract import ROOT_BRANCH_ID
from sts2_training.decision.engine import CombatDecisionEngine
from sts2_training.decision.oracle_log import OracleJsonlWriter, qualified_class_name
from sts2_training.decision.oracle_search import BudgetedOracleCollector, OracleCollectionConfig
from sts2_training.runner._cli import add_common_arguments, configure_logging
from sts2_training.runner.episode import build_engine
from sts2_training.runner.scenario import CombatScenario, EnemyScenario
from sts2_training.selection.heuristic_selector import NoAvailableActionError


@dataclass(frozen=True)
class OracleEpisodeResult:
    instance_id: str
    decisions_collected: int
    final_dto: dict[str, Any]
    elapsed_s: float
    output_path: str


class OracleEpisodeRunner:
    """Drive one started Combat instance while collecting a teacher trace per decision."""

    def __init__(
        self,
        client: Any,
        *,
        oracle: BudgetedOracleCollector,
        commit_engine: CombatDecisionEngine,
        writer: OracleJsonlWriter,
        training_commit: str | None = None,
    ) -> None:
        if commit_engine.client is not client:
            raise ValueError("commit_engine must be bound to the same client")
        self._client = client
        self._oracle = oracle
        self._commit_engine = commit_engine
        self._writer = writer
        self._training_commit = training_commit

    async def run(
        self,
        instance_id: str,
        *,
        oracle_timeout_s: float,
        decision_timeout_s: float,
        max_decisions: int | None = None,
        close_timeout_s: float = 10.0,
    ) -> OracleEpisodeResult:
        """Collect before each commit; ``max_decisions`` is a deliberate collection cap."""

        _positive_timeout("oracle_timeout_s", oracle_timeout_s)
        _positive_timeout("decision_timeout_s", decision_timeout_s)
        _positive_timeout("close_timeout_s", close_timeout_s)
        if max_decisions is not None and (
            isinstance(max_decisions, bool)
            or not isinstance(max_decisions, int)
            or max_decisions <= 0
        ):
            raise ValueError("max_decisions must be a positive integer or None")

        t0 = time.monotonic()
        decisions_collected = 0
        next_decision: dict[str, Any] | None = None
        final_dto: dict[str, Any] = {}
        beam = self._commit_engine.beam_search
        policy_class = qualified_class_name(beam._policy)  # noqa: SLF001
        value_class = qualified_class_name(beam._value_fn)  # noqa: SLF001

        try:
            while True:
                if next_decision is None:
                    raw_decision = await self._client.get_decision(
                        instance_id,
                        ROOT_BRANCH_ID,
                        timeout_s=decision_timeout_s,
                    )
                else:
                    raw_decision = next_decision
                decision = _validated_decision(raw_decision)
                dto = decision["masked_emulator_dto"]
                final_dto = dict(dto)

                if _is_terminal(dto):
                    break
                legal_actions = _available_action_ids(dto)
                if not legal_actions:
                    raise NoAvailableActionError(
                        "non-terminal oracle-collection decision has no legal actions",
                        decision=decision,
                    )

                oracle_result = await self._oracle.collect(
                    instance_id,
                    decision,
                    timeout_s=oracle_timeout_s,
                )
                self._writer.write(
                    decision,
                    oracle_result,
                    policy_class=policy_class,
                    value_class=value_class,
                    training_commit=self._training_commit,
                )
                decisions_collected += 1

                # Commit with the runtime engine, not the teacher, so the state
                # distribution remains aligned with the policy/search we intend to improve.
                outcome = await self._commit_engine.decide(
                    instance_id,
                    timeout_s=decision_timeout_s,
                    decision=decision,
                )
                chosen = outcome.chosen_action_id
                if chosen is None or chosen not in legal_actions:
                    raise NoAvailableActionError(
                        "runtime commit engine did not return an available action",
                        decision=decision,
                    )
                response = await self._client.commit_action(
                    instance_id,
                    decision["decision_point_id"],
                    chosen,
                    timeout_s=decision_timeout_s,
                )
                next_decision = _validated_decision(response)
                final_dto = dict(next_decision["masked_emulator_dto"])
                if _is_terminal(final_dto):
                    break
                if max_decisions is not None and decisions_collected >= max_decisions:
                    break
        finally:
            await self._close_best_effort(instance_id, close_timeout_s)

        return OracleEpisodeResult(
            instance_id=instance_id,
            decisions_collected=decisions_collected,
            final_dto=final_dto,
            elapsed_s=time.monotonic() - t0,
            output_path=str(self._writer.path),
        )

    async def _close_best_effort(self, instance_id: str, timeout_s: float) -> None:
        if getattr(self._client, "pending_retry", None) is not None or getattr(
            self._client, "session_invalid", False
        ):
            return
        try:
            await self._client.close_instance(instance_id, timeout_s=timeout_s)
        except Exception:  # noqa: BLE001 - collection output should survive cleanup failure
            return


def _validated_decision(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("decision/commit response must be a mapping")
    decision_point_id = value.get("decision_point_id")
    dto = value.get("masked_emulator_dto")
    if not isinstance(decision_point_id, str) or not decision_point_id:
        raise RuntimeError("decision is missing decision_point_id")
    if not isinstance(dto, Mapping):
        raise RuntimeError("decision is missing masked_emulator_dto")
    copied = dict(value)
    copied["masked_emulator_dto"] = dict(dto)
    return copied


def _is_terminal(dto: Mapping[str, Any]) -> bool:
    return dto.get("terminal") is True or dto.get("run_terminal") is True


def _available_action_ids(dto: Mapping[str, Any]) -> set[str]:
    actions = dto.get("legal_actions")
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        return set()
    return {
        action_id
        for action in actions
        if isinstance(action, Mapping) and action.get("is_available") is not False
        for action_id in [action.get("action_id")]
        if isinstance(action_id, str) and action_id
    }


def _positive_timeout(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _scenario_from_json(data: dict[str, Any]) -> CombatScenario:
    fields = dict(data)
    fields["enemies"] = [EnemyScenario(**enemy) for enemy in fields["enemies"]]
    return CombatScenario(**fields)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle-beam-width", type=_positive_int, default=32)
    parser.add_argument("--oracle-top-k", type=_positive_int, default=8)
    parser.add_argument("--oracle-depth", type=_positive_int, default=4)
    parser.add_argument("--target-beam-width", type=_positive_int, default=None)
    parser.add_argument("--oracle-timeout", type=_positive_float, default=120.0)
    parser.add_argument("--oracle-time-budget-ms", type=_positive_float, default=None)
    parser.add_argument(
        "--policy-limited-root",
        action="store_true",
        help="do not exhaustively evaluate all root legal actions",
    )
    parser.add_argument("--training-commit", default=None)
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> OracleEpisodeResult:
    scenario = _scenario_from_json(json.loads(args.scenario.read_text(encoding="utf-8")))
    connection = TcpConnection(
        host=args.host,
        port=args.port,
        connect_timeout_s=args.connect_timeout,
    )
    async with AsyncTrainingApiClient(connection) as client:
        commit_engine = build_engine(
            client,
            search_mode=args.search_mode,
            beam_max_depth=args.beam_depth,
        )
        runtime_config = commit_engine.beam_search.config
        oracle_beam_config = replace(
            runtime_config,
            beam_width=args.oracle_beam_width,
            top_k_actions=args.oracle_top_k,
            max_depth=args.oracle_depth,
            time_budget_ms=args.oracle_time_budget_ms,
            simulation_options=(
                None
                if runtime_config.simulation_options is None
                else dict(runtime_config.simulation_options)
            ),
        )
        target_beam_width = (
            runtime_config.beam_width
            if args.target_beam_width is None
            else args.target_beam_width
        )
        oracle = BudgetedOracleCollector.from_beam_engine(
            commit_engine.beam_search,
            config=OracleCollectionConfig(
                beam_config=oracle_beam_config,
                target_beam_width=target_beam_width,
                exhaustive_root_actions=not args.policy_limited_root,
            ),
        )
        instance_id = await client.start_instance(
            scenario.to_instance_config(),
            timeout_s=args.decision_timeout,
        )
        return await OracleEpisodeRunner(
            client,
            oracle=oracle,
            commit_engine=commit_engine,
            writer=OracleJsonlWriter(args.output),
            training_commit=args.training_commit,
        ).run(
            instance_id,
            oracle_timeout_s=args.oracle_timeout,
            decision_timeout_s=args.decision_timeout,
            max_decisions=args.max_decisions,
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging(args.log_level)
    result = asyncio.run(_run(args))
    print(
        json.dumps(
            {
                "instance_id": result.instance_id,
                "decisions_collected": result.decisions_collected,
                "elapsed_s": result.elapsed_s,
                "final_dto": result.final_dto,
                "output_path": result.output_path,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
