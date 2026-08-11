"""Normalized view of the public Combat DTO shared by policy and value code.

Wire/schema interpretation belongs here; policy/value logic should consume this view
instead of independently decoding HP, block, energy, enemy intent, hand/potion metadata,
and powers. `strict=True` preserves ValueModel's input-validation behavior, while the
bootstrap policy uses tolerant normalization so incomplete heuristic metadata simply
removes a bonus rather than making candidate generation fail.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnemyObservation:
    index: int | None
    hp: float
    max_hp: float
    is_alive: bool
    attack_damage: float
    attack_repeats: float
    powers: tuple[Mapping[str, Any], ...]

    @property
    def incoming_attack(self) -> float:
        return max(0.0, self.attack_damage) * max(0.0, self.attack_repeats)


@dataclass(frozen=True)
class CombatObservation:
    hp: float
    max_hp: float
    block: float
    energy: float | None
    enemies: tuple[EnemyObservation, ...]
    hand: tuple[Mapping[str, Any], ...]
    potions: tuple[Mapping[str, Any], ...]
    player_powers: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_dto(
        cls,
        dto: Mapping[str, Any],
        *,
        strict: bool = False,
    ) -> "CombatObservation":
        hp = _number(dto.get("hp"), default=0.0, field="hp", strict=strict)
        max_hp = max(
            1.0,
            _number(dto.get("maxHp"), default=1.0, field="maxHp", strict=strict),
        )
        block = max(
            0.0,
            _number(dto.get("block"), default=0.0, field="block", strict=strict),
        )
        energy = _optional_number(dto.get("energy"), field="energy", strict=strict)

        raw_enemies = _mapping_sequence(dto.get("enemies"), "enemies", strict=strict)
        enemies: list[EnemyObservation] = []
        for position, enemy in enumerate(raw_enemies):
            is_alive_value = enemy.get("isAlive", True)
            if not isinstance(is_alive_value, bool):
                if strict:
                    raise ValueError(
                        f"heuristic input enemies[{position}].isAlive must be a boolean"
                    )
                is_alive_value = True

            intent_value = enemy.get("intent")
            if intent_value is None:
                intent: Mapping[str, Any] = {}
            elif isinstance(intent_value, Mapping):
                intent = intent_value
            elif strict:
                raise ValueError(
                    f"heuristic input enemies[{position}].intent must be a mapping when provided"
                )
            else:
                intent = {}

            damage = _number(
                intent.get("attackDamage"),
                default=0.0,
                field=f"enemies[{position}].intent.attackDamage",
                strict=strict,
            )
            repeats = _number(
                intent.get("attackRepeats"),
                default=1.0,
                field=f"enemies[{position}].intent.attackRepeats",
                strict=strict,
            )
            powers_value = enemy.get("powers")
            if powers_value is None:
                powers_value = enemy.get("enemyPowers")
            powers = tuple(
                _mapping_sequence(
                    powers_value,
                    f"enemies[{position}].powers",
                    strict=strict,
                )
            )
            index_number = _optional_number(
                enemy.get("index"),
                field=f"enemies[{position}].index",
                strict=False,
            )
            enemies.append(
                EnemyObservation(
                    index=int(index_number) if index_number is not None else None,
                    hp=max(
                        0.0,
                        _number(
                            enemy.get("hp"),
                            default=0.0,
                            field=f"enemies[{position}].hp",
                            strict=strict,
                        ),
                    ),
                    max_hp=max(
                        1.0,
                        _number(
                            enemy.get("maxHp"),
                            default=1.0,
                            field=f"enemies[{position}].maxHp",
                            strict=strict,
                        ),
                    ),
                    is_alive=is_alive_value,
                    attack_damage=max(0.0, damage),
                    attack_repeats=max(0.0, repeats),
                    powers=powers,
                )
            )

        return cls(
            hp=max(0.0, hp),
            max_hp=max_hp,
            block=block,
            energy=energy,
            enemies=tuple(enemies),
            hand=tuple(_mapping_sequence(dto.get("hand"), "hand", strict=False)),
            potions=tuple(_mapping_sequence(dto.get("potions"), "potions", strict=False)),
            player_powers=tuple(
                _mapping_sequence(dto.get("playerPowers"), "playerPowers", strict=strict)
            ),
        )

    @property
    def alive_enemies(self) -> tuple[EnemyObservation, ...]:
        return tuple(enemy for enemy in self.enemies if enemy.is_alive)

    @property
    def incoming_before_block(self) -> float:
        return sum(enemy.incoming_attack for enemy in self.alive_enemies)

    @property
    def incoming_damage(self) -> float:
        return max(0.0, self.incoming_before_block - self.block)

    @property
    def danger_ratio(self) -> float:
        return self.incoming_damage / max(1.0, self.hp)

    @property
    def lethal_threat(self) -> bool:
        return self.hp > 0 and self.incoming_damage >= self.hp

    def enemy_by_index(self, index: int) -> EnemyObservation | None:
        return next((enemy for enemy in self.enemies if enemy.index == index), None)


def _mapping_sequence(
    value: Any,
    field: str,
    *,
    strict: bool,
) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        if strict:
            raise ValueError(f"heuristic input {field} must be a sequence of mappings")
        return []
    normalized: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            if strict:
                raise ValueError(f"heuristic input {field}[{index}] must be a mapping")
            continue
        normalized.append(item)
    return normalized


def _optional_number(value: Any, *, field: str, strict: bool) -> float | None:
    if value is None:
        return None
    return _number(value, default=0.0, field=field, strict=strict, missing_is_none=True)


def _number(
    value: Any,
    *,
    default: float,
    field: str,
    strict: bool,
    missing_is_none: bool = False,
) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if strict:
            raise ValueError("heuristic input numbers must be finite numeric values")
        return default
    try:
        number = float(value)
    except OverflowError:
        if strict:
            raise ValueError("heuristic input numbers must be finite numeric values") from None
        return default
    if not math.isfinite(number):
        if strict:
            raise ValueError("heuristic input numbers must be finite numeric values")
        return default
    return number
