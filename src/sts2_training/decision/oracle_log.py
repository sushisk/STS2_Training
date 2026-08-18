"""Stable JSONL records for budgeted Combat oracle collection."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from sts2_training.api.contract import SCHEMA_VERSION
from sts2_training.decision.oracle_search import OracleCollectionResult
from sts2_training.decision.search_trace import BranchFaultTrace, SearchTraceEnd

# v7 extends the persisted search diagnostics from v6 with explicit branch-fault
# observability: search_trace may contain BranchFaultTrace events and search summaries
# carry branches_faulted. The runtime-transition / public DTO contract from v6 is kept.
ORACLE_RECORD_SCHEMA_VERSION = 7
# Episode schema v2 remains stable. ``fault_summary`` is an optional additive diagnostic
# on aborted runs so existing raw-data readers do not need a schema migration.
ORACLE_EPISODE_RESULT_SCHEMA_VERSION = 2
# Value training relies on the full public card-instance identity added by STS2_RL mask
# v1.2: pile multisets retain upgradeLevel, tinker-time state, and enchantment.
ORACLE_VALUE_MASK_VERSION = "1.2"

_STRUCTURAL_FAULT_MARKERS = (
    "snapshot restore rejected",
    "reference_integrity:",
    "dangling reference",
    "missing captured instance",
)
_SETTLEMENT_TIMEOUT_MARKER = "timed out waiting for the next decision point or settlement"


def require_oracle_value_mask_version(
    dto: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Fail closed when a DTO cannot preserve the card identity needed by ValueModel."""

    version = dto.get("mask_version")
    if version != ORACLE_VALUE_MASK_VERSION:
        raise ValueError(
            f"{context} requires masked DTO mask_version={ORACLE_VALUE_MASK_VERSION!r}; "
            f"got {version!r}"
        )


def oracle_value_dto_contract(
    dto: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, str]:
    """Return the public wire identity that gives a Value sample its semantics."""

    require_oracle_value_mask_version(dto, context=context)
    dto_version = dto.get("dto_version")
    if not isinstance(dto_version, str) or not dto_version:
        raise ValueError(f"{context} requires a non-empty masked DTO dto_version")
    return {
        "wire_schema_version": SCHEMA_VERSION,
        "mask_version": ORACLE_VALUE_MASK_VERSION,
        "dto_version": dto_version,
    }


def require_oracle_value_dto_version(
    dto: Mapping[str, Any],
    *,
    expected: str,
    context: str,
) -> None:
    """Require the exact Emulator DTO generation pinned by a learned artifact/dataset."""

    actual = oracle_value_dto_contract(dto, context=context)["dto_version"]
    if actual != expected:
        raise ValueError(
            f"{context} requires masked DTO dto_version={expected!r}; got {actual!r}"
        )


def response_metadata_without_masked_dto(response: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve all public response-envelope fields without duplicating the large DTO."""

    return _jsonable(
        {
            key: value
            for key, value in response.items()
            if key != "masked_emulator_dto"
        }
    )


def _fault_trace_signature(event: BranchFaultTrace) -> str:
    """Return a stable, bounded signature for persisted Oracle fault aggregation."""

    detail = event.detail or ""
    normalized_detail = detail.lower()
    normalized_kind = event.fault_kind.lower() if event.fault_kind else ""
    if normalized_kind in {"reference_integrity", "snapshot_restore"} or any(
        marker in normalized_detail for marker in _STRUCTURAL_FAULT_MARKERS
    ):
        return "snapshot_restore_fault"
    if _SETTLEMENT_TIMEOUT_MARKER in normalized_detail:
        return "settlement_timeout"

    kind = normalized_kind or event.status.lower() or "branch_fault"
    first_line = next(
        (line.strip().lower() for line in detail.splitlines() if line.strip()),
        "",
    )
    if not first_line:
        return kind
    return f"{kind}:{first_line[:160]}"


def _serialized_search_trace(events: Sequence[Any]) -> list[dict[str, Any]]:
    """Serialize trace events while bounding duplicate branch-fault detail volume.

    BranchFaultTrace events remain one-per-logical-fault because Oracle target censoring
    and lineage auditing depend on their branch/RNG locations. For each fault signature,
    only the first persisted trace keeps the full ``detail``. SearchTraceEnd receives a
    bounded aggregate so repeated detail text does not dominate Oracle JSONL size.
    """

    aggregates: dict[str, dict[str, Any]] = {}
    payloads: list[dict[str, Any]] = []
    for event in events:
        if isinstance(event, BranchFaultTrace):
            signature = _fault_trace_signature(event)
            aggregate = aggregates.get(signature)
            first_occurrence = aggregate is None
            if first_occurrence:
                aggregate = {
                    "fault_signature": signature,
                    "fault_kind": event.fault_kind,
                    "count": 0,
                    "first_detail": event.detail,
                    "first_depth": event.combat_depth,
                    "last_depth": event.combat_depth,
                    "first_branch_id": event.branch_id,
                    "last_branch_id": event.branch_id,
                    "action_ids": set(),
                    "action_types": set(),
                    "root_action_ids": set(),
                }
                aggregates[signature] = aggregate
            assert aggregate is not None
            aggregate["count"] += 1
            aggregate["last_depth"] = event.combat_depth
            aggregate["last_branch_id"] = event.branch_id
            aggregate["action_ids"].add(event.action_id)
            if event.action_type is not None:
                aggregate["action_types"].add(event.action_type)
            if event.root_action_id is not None:
                aggregate["root_action_ids"].add(event.root_action_id)

            payload = _jsonable(event)
            if not first_occurrence:
                payload["detail"] = None
            payloads.append(payload)
            continue

        payload = _jsonable(event)
        if isinstance(event, SearchTraceEnd):
            payload["fault_summaries"] = [
                {
                    "fault_signature": aggregate["fault_signature"],
                    "fault_kind": aggregate["fault_kind"],
                    "count": aggregate["count"],
                    "first_detail": aggregate["first_detail"],
                    "first_depth": aggregate["first_depth"],
                    "last_depth": aggregate["last_depth"],
                    "first_branch_id": aggregate["first_branch_id"],
                    "last_branch_id": aggregate["last_branch_id"],
                    "action_ids": sorted(aggregate["action_ids"]),
                    "action_types": sorted(aggregate["action_types"]),
                    "root_action_ids": sorted(aggregate["root_action_ids"]),
                }
                for _, aggregate in sorted(aggregates.items())
            ]
        payloads.append(payload)
    return payloads


def _require_root_value_sample_contracts(samples: Any, *, dto_version: str) -> None:
    if samples is None:
        return
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes, bytearray)):
        raise ValueError("root_value_samples must be a sequence")
    for index, sample in enumerate(samples):
        if isinstance(sample, Mapping):
            dto = sample.get("masked_emulator_dto")
        else:
            dto = getattr(sample, "masked_emulator_dto", None)
        if not isinstance(dto, Mapping):
            raise ValueError(
                f"Oracle root_value_samples[{index}] must contain masked_emulator_dto"
            )
        require_oracle_value_dto_version(
            dto,
            expected=dto_version,
            context=f"Oracle root_value_samples[{index}]",
        )


def _runtime_transition(
    value: Mapping[str, Any],
    *,
    dto_version: str,
) -> dict[str, Any]:
    chosen_action_id = value.get("chosen_action_id")
    if not isinstance(chosen_action_id, str) or not chosen_action_id:
        raise ValueError("runtime_transition.chosen_action_id must be a non-empty string")
    chosen_action = value.get("chosen_action")
    if not isinstance(chosen_action, Mapping):
        raise ValueError("runtime_transition.chosen_action must be an object")
    next_decision_point_id = value.get("next_decision_point_id")
    if not isinstance(next_decision_point_id, str) or not next_decision_point_id:
        raise ValueError(
            "runtime_transition.next_decision_point_id must be a non-empty string"
        )
    next_dto = value.get("next_masked_emulator_dto")
    if not isinstance(next_dto, Mapping):
        raise ValueError("runtime_transition.next_masked_emulator_dto must be an object")
    require_oracle_value_dto_version(
        next_dto,
        expected=dto_version,
        context="Oracle runtime_transition.next_masked_emulator_dto",
    )
    metadata = value.get("commit_response_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("runtime_transition.commit_response_metadata must be an object")
    result = dict(value)
    result["next_dto_contract"] = oracle_value_dto_contract(
        next_dto,
        context="Oracle runtime_transition.next_masked_emulator_dto",
    )
    result["combat_result"] = combat_result_from_dto(next_dto)
    return _jsonable(result)


def _bounded_beam_search_result(value: Any) -> dict[str, Any]:
    """Serialize search diagnostics while omitting deep-node DTO and branch-log payloads."""

    best_node = getattr(value, "best_node", None)
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
            "action": None if best_node.action is None else _jsonable(best_node.action),
            "policy_rank": best_node.policy_rank,
            "policy_score": best_node.policy_score,
            "post_coverage_rank": best_node.post_coverage_rank,
            "candidate_source": best_node.candidate_source,
            "omitted_large_fields": ["masked_emulator_dto", "branch_log"],
        }
    stats = getattr(value, "stats", None)
    return {
        "best_root_action_id": getattr(value, "best_root_action_id", None),
        "best_value": getattr(value, "best_value", None),
        "best_node": best_node_payload,
        "reason": getattr(value, "reason", None),
        "stats": None if stats is None else _jsonable(stats),
    }


def oracle_collection_record(
    root_decision: Mapping[str, Any],
    result: OracleCollectionResult,
    *,
    instance_id: str,
    decision_index: int,
    runtime_transition: Mapping[str, Any],
    training_commit: str | None = None,
) -> dict[str, Any]:
    """Build one self-contained Oracle + actual-runtime record for one root Decision."""

    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("instance_id must be a non-empty string")
    if (
        isinstance(decision_index, bool)
        or not isinstance(decision_index, int)
        or decision_index < 0
    ):
        raise ValueError("decision_index must be a non-negative integer")
    decision_point_id = root_decision.get("decision_point_id")
    dto = root_decision.get("masked_emulator_dto")
    if not isinstance(decision_point_id, str) or not decision_point_id:
        raise ValueError("root_decision must contain a non-empty decision_point_id")
    if not isinstance(dto, Mapping):
        raise ValueError("root_decision must contain masked_emulator_dto")
    dto_contract = oracle_value_dto_contract(dto, context="Oracle decision log")
    root_value_samples = getattr(result, "root_value_samples", ())
    _require_root_value_sample_contracts(
        root_value_samples,
        dto_version=dto_contract["dto_version"],
    )
    transition = _runtime_transition(
        runtime_transition,
        dto_version=dto_contract["dto_version"],
    )

    return {
        "record_type": "combat_oracle_decision",
        "record_schema_version": ORACLE_RECORD_SCHEMA_VERSION,
        "instance_id": instance_id,
        "decision_index": decision_index,
        "decision_point_id": decision_point_id,
        "dto_contract": dto_contract,
        # Preserve public response-envelope fields by default. The masked DTO is stored
        # separately so this does not duplicate the largest field.
        "decision_response_metadata": response_metadata_without_masked_dto(root_decision),
        # Keep the raw masked public DTO so future feature extractors can be rebuilt
        # without replaying the emulator or trusting today's feature schema.
        "masked_emulator_dto": _jsonable(dto),
        # Root-only ValueModel data. Deeper Beam branch DTOs remain intentionally absent.
        "root_value_samples": _jsonable(root_value_samples),
        # Search outcome metadata is useful, but a raw BeamSearchResult.best_node contains
        # masked_emulator_dto/branch_log. Keep a bounded summary instead so deep Branch DTOs
        # cannot leak back into the root-only logging contract.
        "oracle_search_result": _bounded_beam_search_result(result.search_result),
        "oracle_targets": _jsonable(result.targets),
        "search_trace": _serialized_search_trace(result.trace),
        # The actual action committed by the runtime policy/search, plus the exact public
        # post-commit DTO. This is distinct from counterfactual Oracle root samples and
        # makes terminal-return/TD Value learning possible without guessing which branch
        # was really taken.
        "runtime_transition": transition,
        "provenance": {
            "training_commit": training_commit,
            "teacher_policy_class": result.provenance.teacher_policy_class,
            "teacher_inner_policy_class": result.provenance.teacher_inner_policy_class,
            "teacher_coverage_policy_class": (
                result.provenance.teacher_coverage_policy_class
            ),
            "teacher_value_class": result.provenance.teacher_value_class,
            "teacher_policy_metadata": _jsonable(
                result.provenance.teacher_policy_metadata
            ),
            "teacher_inner_policy_metadata": _jsonable(
                result.provenance.teacher_inner_policy_metadata
            ),
            "teacher_value_metadata": _jsonable(
                result.provenance.teacher_value_metadata
            ),
            "pruner_name": result.targets.metadata.pruner_name,
            "pruner_version": result.targets.metadata.pruner_version,
            "rng_sampling": result.targets.metadata.rng_sampling,
        },
    }


def combat_result_from_dto(dto: Mapping[str, Any]) -> str | None:
    """Return a normalized terminal Combat result when the public DTO exposes one."""

    outcome = dto.get("outcome")
    if outcome in {"victory", "run_victory"}:
        return "victory"
    if outcome == "defeat":
        return "defeat"
    transition = dto.get("transition")
    if isinstance(transition, Mapping) and transition.get("kind") == "combat_completed":
        victory = transition.get("victory")
        if victory is True:
            return "victory"
        if victory is False:
            return "defeat"
    return None


def oracle_episode_result_record(
    *,
    instance_id: str,
    decisions_collected: int,
    final_dto: Mapping[str, Any],
    final_decision_metadata: Mapping[str, Any],
    completed: bool,
    termination_reason: str,
    elapsed_s: float,
    fault_summary: Any = None,
) -> dict[str, Any]:
    """Build the terminal/truncated episode record appended to the same Oracle JSONL."""

    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("instance_id must be a non-empty string")
    if isinstance(decisions_collected, bool) or not isinstance(decisions_collected, int):
        raise ValueError("decisions_collected must be an integer")
    if decisions_collected < 0:
        raise ValueError("decisions_collected must be non-negative")
    if not isinstance(final_decision_metadata, Mapping):
        raise ValueError("final_decision_metadata must be an object")
    if not isinstance(completed, bool):
        raise TypeError("completed must be a bool")
    if not isinstance(termination_reason, str) or not termination_reason:
        raise ValueError("termination_reason must be a non-empty string")
    if (
        isinstance(elapsed_s, bool)
        or not isinstance(elapsed_s, (int, float))
        or not math.isfinite(float(elapsed_s))
        or elapsed_s < 0
    ):
        raise ValueError("elapsed_s must be a finite non-negative number")
    dto_contract = oracle_value_dto_contract(final_dto, context="Oracle episode result")

    record = {
        "record_type": "combat_oracle_episode_result",
        "record_schema_version": ORACLE_EPISODE_RESULT_SCHEMA_VERSION,
        "instance_id": instance_id,
        "decisions_collected": decisions_collected,
        "completed": completed,
        "termination_reason": termination_reason,
        "combat_result": combat_result_from_dto(final_dto),
        "dto_contract": dto_contract,
        "final_decision_metadata": _jsonable(final_decision_metadata),
        # Persist the final public DTO verbatim so future RL schemas/result fields are not
        # lost merely because today's consumer does not yet use them.
        "final_masked_emulator_dto": _jsonable(final_dto),
        "elapsed_s": float(elapsed_s),
    }
    if fault_summary is not None:
        fault_payload = _jsonable(fault_summary)
        if not isinstance(fault_payload, dict):
            raise ValueError("fault_summary must serialize to an object")
        fault_payload.setdefault("decision_index", decisions_collected)
        record["fault_summary"] = fault_payload
    return record


class OracleJsonlWriter:
    """Append-only writer; intentionally separate from SelectionAudit JSONL."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(
        self,
        root_decision: Mapping[str, Any],
        result: OracleCollectionResult,
        *,
        instance_id: str,
        decision_index: int,
        runtime_transition: Mapping[str, Any],
        training_commit: str | None = None,
    ) -> dict[str, Any]:
        record = oracle_collection_record(
            root_decision,
            result,
            instance_id=instance_id,
            decision_index=decision_index,
            runtime_transition=runtime_transition,
            training_commit=training_commit,
        )
        self._append(record)
        return record

    def write_episode_result(
        self,
        *,
        instance_id: str,
        decisions_collected: int,
        final_dto: Mapping[str, Any],
        final_decision_metadata: Mapping[str, Any],
        completed: bool,
        termination_reason: str,
        elapsed_s: float,
        fault_summary: Any = None,
    ) -> dict[str, Any]:
        record = oracle_episode_result_record(
            instance_id=instance_id,
            decisions_collected=decisions_collected,
            final_dto=final_dto,
            final_decision_metadata=final_decision_metadata,
            completed=completed,
            termination_reason=termination_reason,
            elapsed_s=elapsed_s,
            fault_summary=fault_summary,
        )
        self._append(record)
        return record

    def _append(self, record: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def qualified_class_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not JSON serializable by oracle log contract: {type(value)!r}")