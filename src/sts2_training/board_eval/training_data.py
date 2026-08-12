"""Build Run-level supervised examples from self-play JSONL selection logs."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sts2_training.board_eval.card_features import DEFAULT_CARD_FEATURES_CSV, CardFeatureExtractor
from sts2_training.board_eval.deck_summary import DECK_SUMMARY_FEATURE_NAMES, DeckSummary, summarize_deck
from sts2_training.board_eval.dto_adapter import (
    UnknownCardPolicy,
    board_context_from_dto,
    deck_features_with_unknown_count_from_dto,
)

__all__ = [
    "MODEL_FEATURE_NAMES",
    "NON_DECK_FEATURE_NAMES",
    "NON_DECK_VALUE_FEATURE_NAMES",
    "BoardStateExample",
    "build_examples_from_log",
    "context_model_features",
    "iter_log_events",
    "label_from_events",
]

_LOG = logging.getLogger(__name__)

NON_DECK_VALUE_FEATURE_NAMES: tuple[str, ...] = (
    "hp",
    "max_hp",
    "gold",
    "act_floor",
    "total_floor",
)
NON_DECK_FEATURE_NAMES: tuple[str, ...] = tuple(
    feature_name
    for value_name in NON_DECK_VALUE_FEATURE_NAMES
    for feature_name in (value_name, f"{value_name}_missing")
)
MODEL_FEATURE_NAMES: tuple[str, ...] = NON_DECK_FEATURE_NAMES + tuple(
    f"deck_{name}" for name in DECK_SUMMARY_FEATURE_NAMES
)


@dataclass(frozen=True)
class BoardStateExample:
    run_id: str
    decision_point_id: str | None
    state_kind: str | None
    deck_summary: DeckSummary
    hp: float | None
    max_hp: float | None
    gold: float | None
    act_floor: int | None
    total_floor: int | None
    label: int

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "decision_point_id": self.decision_point_id,
            "state_kind": self.state_kind,
            "label": self.label,
        }
        row.update(
            context_model_features(
                {
                    "hp": self.hp,
                    "max_hp": self.max_hp,
                    "gold": self.gold,
                    "act_floor": self.act_floor,
                    "total_floor": self.total_floor,
                }
            )
        )
        row.update({f"deck_{name}": value for name, value in asdict(self.deck_summary).items()})
        return row

    def to_model_vector(self) -> tuple[float, ...]:
        row = self.to_row()
        return tuple(float(row[name]) for name in MODEL_FEATURE_NAMES)


def context_model_features(values: Mapping[str, object]) -> dict[str, float]:
    """Encode each context value beside an explicit missingness indicator."""
    result: dict[str, float] = {}
    for name in NON_DECK_VALUE_FEATURE_NAMES:
        value, missing = _as_model_float(values.get(name))
        result[name] = value
        result[f"{name}_missing"] = missing
    return result


def iter_log_events(log_path: Path) -> Iterator[dict[str, Any]]:
    with log_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{log_path}:{line_number}: invalid JSON") from exc


def label_from_events(events: Sequence[Mapping[str, Any]]) -> int | None:
    """Read the terminal Run label from current or legacy self-play log records."""
    for event in reversed(events):
        outcome: object | None = None

        # Current runner.self_play appends one explicit terminal record to every
        # successfully and completely logged Run. Prefer that authoritative record.
        if event.get("event") == "self_play_run_result":
            outcome = event.get("outcome")
            if outcome is None:
                final_dto = event.get("final_dto")
                outcome = final_dto.get("outcome") if isinstance(final_dto, Mapping) else None

        if outcome is None:
            run_result = event.get("run_result")
            if isinstance(run_result, Mapping):
                outcome = run_result.get("outcome")
            elif isinstance(run_result, str):
                outcome = run_result

        if outcome is None:
            result = event.get("result")
            dto = result.get("masked_emulator_dto") if isinstance(result, Mapping) else None
            outcome = dto.get("outcome") if isinstance(dto, Mapping) else None

        if outcome == "victory":
            return 1
        if outcome == "defeat":
            return 0
    return None


def _decision_dto(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
    received = event.get("received")
    dto = received.get("masked_emulator_dto") if isinstance(received, Mapping) else None
    return dto if isinstance(dto, Mapping) else None


def build_examples_from_log(
    log_path: Path,
    extractor: CardFeatureExtractor,
    *,
    on_unknown_card: UnknownCardPolicy = "raise",
    state_kinds: Collection[str] | None = None,
) -> list[BoardStateExample]:
    events = list(iter_log_events(log_path))
    label = label_from_events(events)
    if label is None:
        _LOG.info("%s: no terminal outcome found, skipping (0 examples)", log_path)
        return []

    allowed_state_kinds = set(state_kinds) if state_kinds is not None else None
    examples: list[BoardStateExample] = []
    for event in events:
        dto = _decision_dto(event)
        if dto is None or "deck" not in dto:
            continue
        context = board_context_from_dto(dto, event_boundary=event.get("boundary"))
        state_kind = context["state_kind"] if isinstance(context["state_kind"], str) else None
        if allowed_state_kinds is not None and state_kind not in allowed_state_kinds:
            continue

        cards, unknown_count = deck_features_with_unknown_count_from_dto(
            dto,
            extractor,
            on_unknown_card=on_unknown_card,
        )
        if not cards and unknown_count == 0:
            continue
        request = event.get("request")
        decision_point_id = request.get("decision_point_id") if isinstance(request, Mapping) else None
        examples.append(
            BoardStateExample(
                run_id=log_path.stem,
                decision_point_id=decision_point_id if isinstance(decision_point_id, str) else None,
                state_kind=state_kind,
                deck_summary=summarize_deck(cards, unknown_card_count=unknown_count),
                hp=_optional_number(context["hp"]),
                max_hp=_optional_number(context["max_hp"]),
                gold=_optional_number(context["gold"]),
                act_floor=_optional_int(context["act_floor"]),
                total_floor=_optional_int(context["total_floor"]),
                label=label,
            )
        )
    return examples


def _optional_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _as_model_float(value: object) -> tuple[float, float]:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0, 1.0
    return float(value), 0.0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--card-csv", type=Path, default=DEFAULT_CARD_FEATURES_CSV)
    parser.add_argument("--on-unknown-card", choices=["raise", "skip"], default="raise")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    extractor = CardFeatureExtractor.from_csv(args.card_csv)
    log_paths = sorted(args.log_dir.glob("*.jsonl"))
    all_examples: list[BoardStateExample] = []
    for log_path in log_paths:
        all_examples.extend(
            build_examples_from_log(log_path, extractor, on_unknown_card=args.on_unknown_card)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for example in all_examples:
            handle.write(json.dumps(example.to_row(), ensure_ascii=False) + "\n")
    print(f"{len(log_paths)} log file(s) -> {len(all_examples)} example(s) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
