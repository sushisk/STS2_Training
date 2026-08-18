"""Collect budgeted Oracle targets at each root Combat decision.

The Oracle only observes via branch emulation; the committed root action is still chosen
by a normal ``CombatDecisionEngine``. This keeps the collected state distribution tied to
the runtime policy/search configuration instead of silently switching play to the wider
teacher search.

CLI example::

    python -m sts2_training.runner.oracle_collection \
        --scenario combat.json --output data/combat_oracle.jsonl \
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
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from sts2_training.api import AsyncTrainingApiClient, TcpConnection
from sts2_training.api.contract import ROOT_BRANCH_ID
from sts2_training.decision.beam_search import (
    AllBranchesFaultedError,
    BranchFaultAbortError,
    BranchFaultPolicy,
    BranchFaultSummary,
)
from sts2_training.decision.engine import CombatDecisionEngine
from sts2_training.decision.oracle_log import (
    OracleJsonlWriter,
    combat_result_from_dto,
    response_metadata_without_masked_dto,
)
from sts2_training.decision.oracle_search import OracleCollectionConfig
from sts2_training.decision.oracle_value_logging import RootValueLoggingOracleCollector
from sts2_training.runner._cli import add_common_arguments, configure_logging
from sts2_training.runner.episode import build_engine
from sts2_training.runner.scenario import CombatScenario, EnemyScenario
from sts2_training.selection.heuristic_selector import NoAvailableActionError


@dataclass(frozen=True)
class OracleEpisodeResult:
    instance_id: str
    decisions_collected: int
    final_dto: dict[str, Any]
    combat_result: str | None
    completed: bool
    termination_reason: str
    elapsed_s: float
    output_path: str
    fault_summary: BranchFaultSummary | None = None


# //WORKING
# 現在の実装:
# - BeamSearchConfig.branch_fault_policy=None は既存 runtime Beam の retry 挙動を維持し、
#   Oracle CLI の teacher Beam config だけ BranchFaultPolicy() を有効化する。
# - fault signature は decision/branch_faults.py に集約し、Beam policy と Oracle JSONL が
#   同じ classifier を使う。Snapshot restore/reference_integrity/dangling/missing instance と、
#   STS2_RL の typed snapshot_restore_missing_monster_move を structural として即 abort する。
# - この Runner が BranchFaultAbortError を runtime decide/commit 前に捕捉するため、該当 decision の
#   runtime action は commit されず、episode-result v2 の optional fault_summary に1件だけ残す。
#   永続化時には decisions_collected を zero-based decision_index として summary に付与する。
# - settlement timeout は logical branch ごとの timeout 発生回数で初回+retry 1回まで。
#   global max_branch_attempts は別上限として維持するため generic -> timeout -> success でも
#   timeout 固有の retry を失わない。generic fault 自体は従来の3 attemptを維持する。
# - timeout frontier budget へ数えるのは timeout 自体を2回観測して retry budget を使い切った
#   persistent logical timeout のみ。generic -> generic -> 初回 timeout が global 3-attempt 上限に
#   当たったケースは final fault にはするが timeout budget へ入れない。これは再レビューで発見・修正済み。
# - persistent timeout が同一元 frontier で count>=3 または10%以上なら
#   aborted_settlement_timeout_budget。RL worker-level task_timeout は settlement timeout と別分類。
# - non-fatal BranchFaultTrace は target censoring/RNG lineage のため in-memory では全 detail を保持し、
#   JSONL だけ同一 signature の full detail を初回に限定して search_end.fault_summaries へ集約する。
# paired 静的照合:
# - STS2_RL main 12bc6176e47b6d5659ef7aab6d73925421803092 は restore 後の alive enemy Move
#   欠落を CLR Step 前に検出し snapshot_restore_missing_monster_move を返す。
# - STS2_Emulator f708ebb650bc159e3afdff891520deb4a0510e89 は EnemySnapshot.Intent/StateLog を
#   capture し、SnapshotRestorer.ApplyEnemyMoves() で stateId/history を戻して
#   MonsterModel.SetMoveImmediate(..., forceTransition=True) 相当で pending Move を明示復元する。
# - よって typed missing-move fault は現行正常 restore の既知 gap ではなく、正常 snapshot なら
#   成立すべき post-restore invariant が崩れた場合を opaque settlement timeout 前に止める defensive
#   structural boundary と判断し、Training では deterministic/non-retryable とする。
# 検証状況:
# - structural/timeout/generic retry、runtime commit 無し abort、schema v2 summary+decision_index、
#   JSONL aggregation、BeamSearch.search() 例外伝播と cancel/release cleanup を回帰テスト化済み。
# - paired tests は typed restore fault 即 abort、generic -> timeout -> success、worker task_timeout 非誤分類、
#   generic -> generic -> first timeout at global limit の timeout-budget 非誤発火まで固定した。
# - latest test head cde4fd94ea4665dfe8666333f58c44477c8d4f48 の training-hosted-contract は success。
#   oracle_log/search_trace/runner の追加差分も再レビューし、現時点で新しい repo-local blocker は未発見。
# 次の作業者向け:
# - 実 STS2_RL/Emulator が使えるなら Defect 系 restore case で typed structural fault 1回の episode abort、
#   runtime commit 無しを paired log で確認する。
# - Regent case では isolated opaque settlement timeout が1 retryで回復可能であること、persistent な
#   場合だけ count/ratio budget で abort することを実ログ確認する。
# - generic fault の console warning は意図的に既存のまま。対象 fatal structural/timeout cascade は
#   _score_frontier の per-branch warning 前に fail-fast/budget abort するため logging filter は追加しない。
class OracleEpisodeRunner:
    """Drive one started Combat instance while collecting a teacher trace per decision."""

    def __init__(
        self,
        client: Any,
        *,
        oracle: RootValueLoggingOracleCollector,
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
        """Collect Oracle labels, then commit runtime actions and log the actual trajectory."""

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
        final_decision_metadata: dict[str, Any] = {}
        completed = False
        termination_reason = "unknown"
        fault_summary: BranchFaultSummary | None = None
        output_start_size = self._output_size()

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
                final_decision_metadata = response_metadata_without_masked_dto(decision)

                if _is_terminal(dto):
                    completed = True
                    termination_reason = "terminal"
                    break
                legal_actions = _available_action_ids(dto)
                if not legal_actions:
                    raise NoAvailableActionError(
                        "non-terminal oracle-collection decision has no legal actions",
                        decision=decision,
                    )

                try:
                    oracle_result = await self._oracle.collect(
                        instance_id,
                        decision,
                        timeout_s=oracle_timeout_s,
                    )

                    # Commit with the runtime engine, not the teacher, so the state
                    # distribution remains aligned with the policy/search we intend
                    # to improve.
                    outcome = await self._commit_engine.decide(
                        instance_id,
                        timeout_s=decision_timeout_s,
                        decision=decision,
                    )
                except BranchFaultAbortError as exc:
                    completed = False
                    termination_reason = exc.termination_reason
                    fault_summary = exc.summary
                    break
                except AllBranchesFaultedError:
                    completed = False
                    # "repeated" now means that the per-candidate branch-attempt budget
                    # was exhausted inside BeamSearch, not that the whole search reran.
                    termination_reason = "aborted_repeated_branch_failure"
                    break

                chosen = outcome.chosen_action_id
                if chosen is None or chosen not in legal_actions:
                    raise NoAvailableActionError(
                        "runtime commit engine did not return an available action",
                        decision=decision,
                    )
                chosen_action = _available_action_by_id(dto, chosen)
                if chosen_action is None:
                    raise NoAvailableActionError(
                        "runtime chosen action could not be recovered from public legal_actions",
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
                final_decision_metadata = response_metadata_without_masked_dto(next_decision)

                # Write only after the runtime commit succeeds. Each retained record is a
                # complete (public pre-state, Oracle counterfactuals, actual action, public
                # post-state) tuple. Runtime search diagnostics are retained, but the
                # BeamSearchResult is summarized so its deep best-node DTO/branch_log do
                # not violate the explicit root-only DTO volume boundary.
                self._writer.write(
                    decision,
                    oracle_result,
                    instance_id=instance_id,
                    decision_index=decisions_collected,
                    runtime_transition={
                        "chosen_action_id": chosen,
                        "chosen_action": chosen_action,
                        "decision_source": outcome.source,
                        "beam_result": _beam_result_summary(outcome.beam_result),
                        "next_decision_point_id": next_decision["decision_point_id"],
                        "commit_response_metadata": final_decision_metadata,
                        "next_masked_emulator_dto": final_dto,
                    },
                    training_commit=self._training_commit,
                )
                decisions_collected += 1

                if _is_terminal(final_dto):
                    completed = True
                    termination_reason = "terminal"
                    break
                if max_decisions is not None and decisions_collected >= max_decisions:
                    termination_reason = "max_decisions"
                    break

            # The episode-result record is part of the same atomic append as its decision
            # records. If constructing or appending it fails (including a partial-line I/O
            # failure), the outer rollback below truncates everything written by this
            # episode back to the exact pre-episode byte size.
            elapsed_s = time.monotonic() - t0
            self._writer.write_episode_result(
                instance_id=instance_id,
                decisions_collected=decisions_collected,
                final_dto=final_dto,
                final_decision_metadata=final_decision_metadata,
                completed=completed,
                termination_reason=termination_reason,
                elapsed_s=elapsed_s,
                fault_summary=fault_summary,
            )
            return OracleEpisodeResult(
                instance_id=instance_id,
                decisions_collected=decisions_collected,
                final_dto=final_dto,
                combat_result=combat_result_from_dto(final_dto),
                completed=completed,
                termination_reason=termination_reason,
                elapsed_s=elapsed_s,
                output_path=str(self._writer.path),
                fault_summary=fault_summary,
            )
        except Exception as exc:
            try:
                self._rollback_output(output_start_size)
            except Exception as rollback_exc:
                raise ExceptionGroup(
                    "Oracle collection failed and partial JSONL rollback also failed",
                    [exc, rollback_exc],
                ) from None
            raise
        finally:
            await self._close_best_effort(instance_id, close_timeout_s)

    def _output_size(self) -> int:
        try:
            return self._writer.path.stat().st_size
        except FileNotFoundError:
            return 0

    def _rollback_output(self, start_size: int) -> None:
        path = self._writer.path
        try:
            current_size = path.stat().st_size
        except FileNotFoundError:
            if start_size == 0:
                return
            raise RuntimeError("Oracle output disappeared before rollback") from None
        if current_size < start_size:
            raise RuntimeError(
                "Oracle output shrank below its pre-episode size; refusing unsafe rollback"
            )
        with path.open("r+b") as handle:
            handle.truncate(start_size)

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
    if dto.get("terminal") is True or dto.get("run_terminal") is True:
        return True
    boundary = dto.get("boundary")
    if isinstance(boundary, str) and boundary in {"terminal", "run_terminal"}:
        return True
    transition = dto.get("transition")
    return isinstance(transition, Mapping) and transition.get("kind") == "combat_completed"


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


def _available_action_by_id(dto: Mapping[str, Any], action_id: str) -> dict[str, Any] | None:
    actions = dto.get("legal_actions")
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        return None
    for action in actions:
        if (
            isinstance(action, Mapping)
            and action.get("is_available") is not False
            and action.get("action_id") == action_id
        ):
            return dict(action)
    return None


def _beam_result_summary(result: Any) -> dict[str, Any] | None:
    """Preserve runtime search diagnostics without serializing a deep Beam DTO."""

    if result is None:
        return None
    stats = getattr(result, "stats", None)
    stats_payload = None if stats is None else dict(vars(stats))
    best_node = getattr(result, "best_node", None)
    best_node_payload: dict[str, Any] | None = None
    if best_node is not None:
        best_node_payload = {
            "branch_id": best_node.branch_id,
            "parent_branch_id": best_node.parent_branch_id,
            "rng_id": best_node.rng_id,
            "decision_point_id": best_node.decision_point_id,
            "depth": best_node.depth,
            "value": best_node.value,
            "root_action_id": best_node.root_action_id,
            "combat_depth": best_node.combat_depth,
            "continuation_steps": best_node.continuation_steps,
            "terminal": best_node.terminal,
            "action_id": best_node.action_id,
            "action_type": best_node.action_type,
            "action": None if best_node.action is None else dict(best_node.action),
            "policy_rank": best_node.policy_rank,
            "policy_score": best_node.policy_score,
            "post_coverage_rank": best_node.post_coverage_rank,
            "candidate_source": best_node.candidate_source,
            "omitted_large_fields": ["masked_emulator_dto", "branch_log"],
        }
    return {
        "best_root_action_id": getattr(result, "best_root_action_id", None),
        "best_value": getattr(result, "best_value", None),
        "best_node": best_node_payload,
        "reason": getattr(result, "reason", None),
        "stats": stats_payload,
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
            branch_fault_policy=BranchFaultPolicy(),
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
        oracle = RootValueLoggingOracleCollector.from_beam_engine(
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
                "combat_result": result.combat_result,
                "completed": result.completed,
                "termination_reason": result.termination_reason,
                "fault_summary": (
                    None if result.fault_summary is None else asdict(result.fault_summary)
                ),
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