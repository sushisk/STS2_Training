"""Extract per-card secondary numeric features (attack/defense/draw/energy/
power application/...) from STS2_Emulator's decompiled card source, to gauge
how far automatic extraction can go before deck-evaluation features need
hand-authoring.

Source of truth: `Sts2Emulator/Imported/Source/MegaCrit.Sts2.Core.Models.Cards/
*.cs`. Each card declares its scalable numeric effects through a
`CanonicalVars` property, e.g.:

    protected override IEnumerable<DynamicVar> CanonicalVars => ... new DynamicVar[2]
    {
        new DamageVar(9m, ValueProp.Move),
        new CardsVar(1)
    };

    protected override IEnumerable<DynamicVar> CanonicalVars => ...
        new PowerVar<WeakPower>(1m);

This script does NOT execute or fully parse C# - it locates each card's
`CanonicalVars` expression body (via brace/paren-balanced scanning, since the
body can span one line or many) and regex-matches `new XxxVar(...)` /
`new XxxVar<PowerClass>(...)` / `new DynamicVar("Label", value)` calls inside
it. `PowerVar<T>` magnitudes are additionally cross-referenced against
`MegaCrit.Sts2.Core.Models.Powers/*.cs` (which declares each Power's
`PowerType.Buff`/`PowerType.Debuff`/... via a one-line property override) and
against `Common/ids/powers.json` (class name -> power id), so a card that
applies e.g. Weak comes out as a classified Debuff application, not just an
opaque number.

This only recovers *declared* base values - conditional/state-dependent
effects (e.g. "deal damage equal to your Strength") live in each card's
`OnPlay` method body as real C# logic and are NOT extracted here; those, and
any card this script marks non-"auto", still need hand authoring. The report
below exists to measure how much of that hand-authoring burden actually
remains.

Usage:
    python tools/extract_card_effects.py
    python tools/extract_card_effects.py --rl-root D:\\STS2_RL --emulator-root D:\\STS2_Emulator
    python tools/extract_card_effects.py --force   # overwrite a previously hand-edited CSV

By default this writes ONE human-editable CSV (`--csv`, default
`tools/output/card_secondary_features.csv`) meant to be hand-patched
afterward (fill in state-dependent/`no_canonical_vars` rows, correct
anything the regex extraction got wrong) and then used as the actual input to
deck evaluation - NOT meant to be silently regenerated over existing edits,
so by default this script REFUSES to overwrite an existing `--csv` file; pass
`--force` to intentionally regenerate it from scratch. The full raw JSON dump
(`--output`, off by default) is a separate, disposable debugging artifact
that IS always safe to overwrite - it is not meant to be hand-edited.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_RL_ROOT = Path(os.environ.get("STS2_RL_ROOT", r"C:\STS2_RL"))
DEFAULT_EMULATOR_ROOT = Path(os.environ.get("STS2_EMULATOR_ROOT", r"C:\STS2_Emulator"))

CARDS_JSON_RELATIVE = Path("Common/ids/cards.json")
POWERS_JSON_RELATIVE = Path("Common/ids/powers.json")
CARD_SOURCE_RELATIVE = Path("Sts2Emulator/Imported/Source/MegaCrit.Sts2.Core.Models.Cards")
POWER_SOURCE_RELATIVE = Path("Sts2Emulator/Imported/Source/MegaCrit.Sts2.Core.Models.Powers")

# Curse/Status cards are structurally hindrance cards (some do have real
# effects, but they are never something a deck is built around) - excluded
# from normal per-cost deck evaluation by default, not folded into the
# regular secondary-feature averages.
_CURSE_STATUS_TYPES = frozenset({"Curse", "Status"})
# Non-Curse/Status cards judged functionally curse-equivalent by manual
# review (e.g. unplayable Quest items) - same exclusion treatment.
_CURSE_EQUIVALENT_CARD_IDS = frozenset({"BYRDONIS_EGG", "LANTERN_KEY"})

# User-specified direct overrides where an automatic numeric estimate is
# either impossible (state-doubling effects) or was explicitly given a
# substitute number during manual review - NOT inferred, always literal
# instructions.
_MANUAL_FEATURE_OVERRIDES: dict[str, dict[str, float]] = {
    # DoubleEnergy.cs: PlayerCmd.GainEnergy(currentEnergy) - doubles current
    # energy. Flat approximation per manual review instruction.
    "DOUBLE_ENERGY": {"energy": 1.5},
}
_MANUAL_POWER_OVERRIDES: dict[str, list[dict[str, Any]]] = {
    # Scare.cs: PowerCmd.Apply<WeakPower>(..., 1m, ...) per hittable enemy -
    # a literal magnitude outside CanonicalVars (same shape Plan D would
    # cover generally; hand-verified against source for this one card only).
    "SCARE": [{"power_class": "WeakPower", "base_value": 1.0}],
}

# Var type name (without the trailing "Var") -> named secondary feature it
# contributes to. Extend this as more Var types turn out to matter for deck
# evaluation; anything NOT listed here still gets extracted (see
# `recognized_vars` in the report) but does not roll up into a named feature.
_VAR_FEATURE_MAP: dict[str, str] = {
    "Damage": "damage",
    "CalculatedDamage": "damage",
    "OstyDamage": "damage",
    "ExtraDamage": "extra_damage",
    "Block": "block",
    "CalculatedBlock": "block",
    "Cards": "cards_drawn",
    "Energy": "energy",
    "Repeat": "repeat_hits",
    "Heal": "heal",
    "Gold": "gold",
    "MaxHp": "max_hp",
    "HpLoss": "hp_loss",
    "Forge": "forge",
    "Stars": "stars",
    "Summon": "summon",
}

_CLASS_RE = re.compile(r"public\s+sealed\s+class\s+(\w+)\s*:\s*CardModel")
_BASE_CTOR_RE = re.compile(
    r":\s*base\s*\(\s*(-?\d+)\s*,\s*CardType\.(\w+)\s*,\s*CardRarity\.(\w+)\s*,\s*TargetType\.(\w+)\s*\)"
)
_CANONICAL_VARS_RE = re.compile(r"CanonicalVars\s*=>")
_POWER_TYPE_RE = re.compile(r"PowerType\s+Type\s*=>\s*PowerType\.(\w+)")

# One regex covering every observed `new ...Var(...)` shape:
#   new DamageVar(9m, ValueProp.Move)
#   new PowerVar<WeakPower>(1m)
#   new DynamicVar("Accelerant", 1m)          <- opaque, label + value
#   new DynamicVar[2] { ... }                 <- array wrapper, not matched itself
_VAR_CALL_RE = re.compile(
    r"new\s+(?P<name>\w+?)Var(?:<(?P<generic>\w+)>)?\s*\(\s*"
    r'(?:"(?P<label>[^"]*)"\s*,\s*)?'
    r"(?P<value>-?\d+(?:\.\d+)?)m?"
)
# Broad net used only to sanity-check _VAR_CALL_RE didn't miss anything. Must
# require an opening paren (constructor call), NOT `[` - `new DynamicVar[2]
# { ... }` is the array-literal wrapper most multi-var CanonicalVars bodies
# use, not a Var instance itself, and must not be counted as one.
_VAR_CALL_BROAD_RE = re.compile(r"new\s+\w+Var(?:<\w+>)?\s*\(")

# Card generation / transform / orb-channel mechanics live in OnPlay (etc.) as
# real method calls, NOT inside CanonicalVars, so these are scanned over the
# WHOLE file separately from _parse_canonical_vars. `CreateCard<T>()` is the
# one shared primitive for both "generate a new card" and "transform into a
# new card" (Transform() is called on its result when it's a transform);
# OrbCmd.Channel<T>() is the orb-generation primitive.
_CREATE_CARD_RE = re.compile(r"CreateCard<(\w+)>\s*\(")
_TRANSFORM_CALL_RE = re.compile(r"CardCmd\.Transform\s*\(")
_ORB_CHANNEL_RE = re.compile(r"OrbCmd\.Channel<(\w+)>\s*\(")
# Soul (Necrobinder token card) is generated through its own static helper
# instead of the shared CreateCard<T>() primitive - `Soul.Create(...)` /
# `Soul.CreateInHand(...)` - and can also be produced by transforming an
# existing card via `CardCmd.TransformTo<T>()` (a second transform primitive
# alongside `CardCmd.Transform()` + CreateCard<T>()). Both were previously
# invisible to card_generation/card_transform entirely.
_SOUL_CREATE_RE = re.compile(r"Soul\.Create(?:InHand)?\s*\(")
_TRANSFORM_TO_RE = re.compile(r"CardCmd\.TransformTo<(\w+)>\s*\(")
# A simple, literal-bounded counted loop immediately preceding a match, e.g.
# `for (int i = 0; i < 2; i++) { ... CreateCard<Wound>(...) ... }` - used as a
# best-effort multiplier; anything else (a variable bound, a `foreach`) is
# left at count=1 and flagged `count_uncertain` for manual review instead of
# guessed at.
_LITERAL_FOR_LOOP_RE = re.compile(r"for\s*\(\s*int\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*(\d+)\s*;")
_DYNAMIC_LOOP_RE = re.compile(r"for\s*\(\s*int\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*(?!\d)\w|foreach\s*\(")
_LOOP_LOOKBEHIND_WINDOW = 200

# "Offer N candidates (or a single automatic pick) from a card pool, filtered
# by CardType.X or a named CardPoolModel, then add the chosen one to hand" -
# CardFactory.Get(Distinct)?ForCombat(...) is the shared primitive across
# Discovery/Quasar/Splash/Abundance/Distraction/InfernalBlade/WhiteNoise/
# Stoke/StormOfSteel-shaped cards. The specific generated card can't be known
# ahead of time (RNG/player choice), so this is recorded WITHOUT a
# created_card_id, tagged only by the pool/type filter - downstream deck eval
# should price it at count * average(score of cards matching that filter).
_CARD_POOL_GENERATION_RE = re.compile(r"CardFactory\.Get(?:Distinct)?ForCombat\(")
_TYPE_FILTER_RE = re.compile(r"c\.Type\s*==\s*CardType\.(\w+)")
_NAMED_POOL_RE = re.compile(r"CardPool<(\w+)>\(\)")
_COUNT_FROM_COLLECTION_RE = re.compile(r"\.Count(?:\(\))?\s*;")
_FORWARD_LOOKAHEAD_WINDOW = 400

# "Search a pile (typically the draw pile) for a card, filtered by type, and
# move it straight to hand" (Wish/SecretTechnique/SecretWeapon) - a
# tutor/search effect over EXISTING deck cards, distinct from generating a
# brand new one.
_TUTOR_SELECT_RE = re.compile(r"CardSelectCmd\.From(?:CombatPile|Hand)\(")

# CanonicalKeywords (Exhaust/Retain/Innate/Ethereal/Unplayable/...) uses the
# same expression-bodied-property shape as CanonicalVars, so it's located and
# balanced-scanned the same way (see _extract_balanced_expression).
_CANONICAL_KEYWORDS_RE = re.compile(r"CanonicalKeywords\s*=>")
_KEYWORD_RE = re.compile(r"CardKeyword\.(\w+)")

# CanonicalTags (Strike/Shiv/Defend/OstyAttack/Minion/...) - same
# expression-bodied-property shape as CanonicalKeywords, and the PRODUCER
# side of tribal-count synergy (a deck's number of Strike cards, etc.).
_CANONICAL_TAGS_RE = re.compile(r"CanonicalTags\s*=>")
_CARD_TAG_RE = re.compile(r"CardTag\.(\w+)")
# CONSUMER side of the same synergy: code that counts/filters cards by tag
# (`....Tags.Contains(CardTag.Strike)`) or, for Soul specifically (not a
# CardTag, a distinct card class), a literal `is Soul` type check.
_TAG_REFERENCE_RE = re.compile(r"Tags\.Contains\(CardTag\.(\w+)\)")
_SOUL_TYPE_CHECK_RE = re.compile(r"\bis\s+Soul\b")

# Exhaust/discard pile-size synergy. CONSUMER: code that counts cards
# currently sitting in the Exhaust/Discard pile (two call shapes seen in the
# source: `PileType.X.GetPile(...).Cards.Count(...)` and
# `CardPile.GetCards(..., PileType.X).Count()`). PRODUCER: on top of the
# existing Exhaust/Retain keyword (self-exhaust, already in `keywords`),
# `CardCmd.Exhaust(...)`/`CardCmd.Discard(...)` explicitly send OTHER cards
# to that pile, which is the more direct fodder-generation signal for this
# synergy.
_PILE_COUNT_RE = re.compile(
    r"PileType\.(Exhaust|Discard)\.GetPile\([^)]*\)\.Cards\.Count"
    r"|CardPile\.GetCards\([^)]*PileType\.(Exhaust|Discard)\)\.Count"
)
_EXHAUST_ACTION_RE = re.compile(r"CardCmd\.Exhaust\s*\(")
_DISCARD_ACTION_RE = re.compile(r"CardCmd\.Discard(?:AndDraw)?\s*\(")

_HAS_ENERGY_COST_X_RE = re.compile(r"HasEnergyCostX\s*=>\s*true")
_MULTIPLAYER_ONLY_RE = re.compile(r"MultiplayerConstraint\s*=>\s*CardMultiplayerConstraint\.MultiplayerOnly")
_STAR_COST_RE = re.compile(r"CanonicalStarCost\s*=>\s*(\d+)")

# Powers with no card-side producer at all (granted only by relics/potions/
# character passives) - excluded from synergy scoring per manual review,
# since a supply/demand pairing is meaningless when supply is structurally
# always zero on the card side.
_SYNERGY_EXCLUDED_POWER_CLASSES = frozenset({"ArtifactPower"})

# Synergy "consumer/payoff" side: a card whose logic reads an EXISTING Power
# already present on a target (usually an enemy debuff like Vulnerable/
# Poison, occasionally a self buff) via GetPower<T>()/GetPowerAmount<T>()/
# HasPower<T>() - this is the counterpart to a "producer" card that applies
# that same Power (see `powers_applied`), and is what makes e.g. an
# Ironclad Vulnerable-application card and a Vulnerable-scaling damage card
# synergistic as a PAIR, not just individually valuable. The receiver
# expression (whatever is left of the `.`) is classified "self" if it
# mentions `Owner`, otherwise "enemy" (best-effort heuristic, not a real
# type check - `target`, `cardPlay.Target`, `e`/`enemy` lambda params,
# `item.Player`, etc. all read as "enemy").
_POWER_REFERENCE_RE = re.compile(
    r"(?P<receiver>[\w.]+)\??\.(?:GetPowerAmount|GetPower|HasPower)<(?P<power_class>\w+)>"
)
# Orb consumption (the counterpart to OrbCmd.Channel<T>() production) -
# EvokeNext triggers an orb's evoke effect early/extra, the payoff side of
# the Defect orb-generation engine.
_ORB_EVOKE_RE = re.compile(r"OrbCmd\.EvokeNext\s*\(")


@dataclass
class CardExtraction:
    card_id: str
    class_name: str
    energy_cost: int | None
    rarity: str | None
    card_type: str | None
    target_type: str | None
    status: str  # "auto" | "no_canonical_vars" | "unparsed_residue" | "no_source_file"
    features: dict[str, float] = field(default_factory=dict)
    powers_applied: list[dict[str, Any]] = field(default_factory=list)
    opaque_values: dict[str, float] = field(default_factory=dict)
    recognized_var_types: list[str] = field(default_factory=list)
    unparsed_snippet: str | None = None
    card_generation: list[dict[str, Any]] = field(default_factory=list)
    card_transform: list[dict[str, Any]] = field(default_factory=list)
    orbs_channeled: list[dict[str, Any]] = field(default_factory=list)
    # Every Power-type card grants SOME ongoing state, even a wholly bespoke
    # one this script cannot quantify (e.g. Barricade's "Block no longer
    # decays") - counted as one flat "applies a state" signal so a Power card
    # is never treated as zero-signal just because its specific effect isn't
    # numerically extractable. Granular buff/debuff magnitudes (via
    # `powers_applied`) are kept separately wherever they ARE extractable;
    # this is a floor, not a replacement for that.
    applies_power_state: bool = False
    # Curse/Status/multiplayer-only cards are out of scope for normal deck
    # evaluation - always populated, never inferred from signal.
    # `exclusion_reason` is "curse_status" | "multiplayer_only" | "".
    deck_eval_excluded: bool = False
    exclusion_reason: str = ""
    keywords: list[str] = field(default_factory=list)
    is_x_cost: bool = False
    star_cost: int | None = None
    # "Propose N (or auto-pick 1) from a filtered card pool" - see
    # _CARD_POOL_GENERATION_RE. Each entry: {"pool_filter": str, "count": int,
    # "count_uncertain": bool}. `created_card_id` is never known here (RNG/
    # player choice), unlike `card_generation`'s CreateCard<T>() entries.
    pool_generation: list[dict[str, Any]] = field(default_factory=list)
    # "Search an existing pile (usually the draw pile) for a card and move it
    # to hand" - see _TUTOR_SELECT_RE. Each entry: {"pool_filter": str,
    # "count": int}.
    card_tutor: list[dict[str, Any]] = field(default_factory=list)
    # Synergy "payoff" side - see _scan_synergy_consumers. Each entry:
    # {"power_class": str, "power_id": str|None, "power_type": str,
    # "referent": "self"|"enemy"}. `orb_evoked` is a plain occurrence count
    # (OrbCmd.EvokeNext calls), the payoff counterpart to `orbs_channeled`.
    synergy_consumes: list[dict[str, Any]] = field(default_factory=list)
    orb_evoked: int = 0
    # Tribal tag synergy (Strike/Shiv/Defend/OstyAttack/Minion/Soul) - see
    # _scan_card_tags (producer) / _scan_tribal_references (consumer).
    card_tags: list[str] = field(default_factory=list)
    tribal_references: list[str] = field(default_factory=list)
    # Exhaust/discard pile-size synergy - see _scan_pile_synergy.
    exhaust_pile_reference: bool = False
    discard_pile_reference: bool = False
    exhausts_other_cards: bool = False
    discards_other_cards: bool = False
    # Sly (CardKeyword.Sly) - a THIRD, distinct kind of discard interaction:
    # not "how many cards are in the discard pile" but "this specific card
    # auto-plays itself when it lands in the discard pile" - a discard
    # payoff, not a pile-size counter. Already present in `keywords` (see
    # _scan_keywords), pulled out here as its own flag purely for visibility
    # in deck-eval synergy grouping.
    sly_keyword: bool = False


def _extract_balanced_expression(text: str, start: int) -> str:
    """Returns the substring from `start` up to (excluding) the top-level
    `;` that closes an expression-bodied member, tracking (), {}, [] depth so
    semicolons inside nested constructs don't end the scan early."""
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth -= 1
        elif char == ";" and depth == 0:
            return text[start:index]
    return text[start:]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _build_power_type_by_class(power_source_dir: Path) -> dict[str, str]:
    power_type_by_class: dict[str, str] = {}
    for cs_file in power_source_dir.glob("*.cs"):
        text = cs_file.read_text(encoding="utf-8")
        class_match = re.search(r"public\s+sealed\s+class\s+(\w+)\s*:\s*PowerModel", text)
        type_match = _POWER_TYPE_RE.search(text)
        if class_match and type_match:
            power_type_by_class[class_match.group(1)] = type_match.group(1)
    return power_type_by_class


def _parse_canonical_vars(body: str) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, float], list[str]]:
    """Returns (typed_feature_sums, power_applications, opaque_labeled_values,
    recognized_var_type_names) for one card's CanonicalVars expression body."""
    typed_features: dict[str, float] = {}
    powers_applied: list[dict[str, Any]] = []
    opaque_values: dict[str, float] = {}
    recognized_types: list[str] = []

    for match in _VAR_CALL_RE.finditer(body):
        name = match.group("name")
        generic = match.group("generic")
        label = match.group("label")
        value = float(match.group("value"))

        if name == "Dynamic":
            key = label or f"unlabeled_{len(opaque_values)}"
            opaque_values[key] = value
            recognized_types.append("DynamicVar")
            continue

        var_type = f"{name}Var"
        recognized_types.append(var_type)

        if name == "Power" and generic:
            powers_applied.append({"power_class": generic, "base_value": value})
            continue

        feature_name = _VAR_FEATURE_MAP.get(name)
        if feature_name is not None:
            typed_features[feature_name] = typed_features.get(feature_name, 0.0) + value
        else:
            opaque_values[var_type] = opaque_values.get(var_type, 0.0) + value

    return typed_features, powers_applied, opaque_values, recognized_types


def _estimate_count(text: str, match_start: int) -> tuple[int, bool]:
    """Best-effort repetition count for a `CreateCard<T>(`/`OrbCmd.Channel<T>(`
    occurrence at `match_start`: looks for a simple literal-bounded `for` loop
    in the preceding `_LOOP_LOOKBEHIND_WINDOW` characters. Returns
    `(count, count_uncertain)` - `count_uncertain=True` means a loop was
    detected but its bound isn't a plain literal (variable, or `foreach`), so
    the true count could be anything and needs manual review rather than a
    guessed number."""
    window_start = max(0, match_start - _LOOP_LOOKBEHIND_WINDOW)
    window = text[window_start:match_start]
    literal_matches = list(_LITERAL_FOR_LOOP_RE.finditer(window))
    if literal_matches:
        return int(literal_matches[-1].group(1)), False
    if _DYNAMIC_LOOP_RE.search(window):
        return 1, True
    return 1, False


def _scan_whole_file_mechanics(
    text: str, card_id_by_class: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Card generation, card transform, and orb-channel mechanics all live in
    real method bodies (typically `OnPlay`), not in `CanonicalVars`, so this
    scans the whole card source file rather than one expression body. A card
    whose file contains `CardCmd.Transform(` is treated as using EVERY
    `CreateCard<T>()` in that file as a transform target; otherwise each
    `CreateCard<T>()` is treated as pure generation. This is a per-file, not
    per-call-site, classification - a card mixing both generation and
    transform in one file would be misclassified, but no such card was found
    while building this."""
    is_transform_card = _TRANSFORM_CALL_RE.search(text) is not None

    generation: list[dict[str, Any]] = []
    transform: list[dict[str, Any]] = []
    for match in _CREATE_CARD_RE.finditer(text):
        class_name = match.group(1)
        count, uncertain = _estimate_count(text, match.start())
        entry = {
            "created_class": class_name,
            "created_card_id": card_id_by_class.get(class_name),
            "count": count,
            "count_uncertain": uncertain,
        }
        (transform if is_transform_card else generation).append(entry)

    # Soul.Create(InHand)(...) never carries a plain literal count argument
    # in practice (always a variable/DynamicVars read), so this is always
    # recorded as count=1/uncertain=True rather than guessed at.
    for _ in _SOUL_CREATE_RE.finditer(text):
        generation.append(
            {
                "created_class": "Soul",
                "created_card_id": card_id_by_class.get("Soul"),
                "count": 1,
                "count_uncertain": True,
            }
        )
    for match in _TRANSFORM_TO_RE.finditer(text):
        class_name = match.group(1)
        count, uncertain = _estimate_count(text, match.start())
        transform.append(
            {
                "created_class": class_name,
                "created_card_id": card_id_by_class.get(class_name),
                "count": count,
                "count_uncertain": uncertain,
            }
        )

    orbs: list[dict[str, Any]] = []
    for match in _ORB_CHANNEL_RE.finditer(text):
        orb_class = match.group(1)
        count, uncertain = _estimate_count(text, match.start())
        orbs.append({"orb_class": orb_class, "count": count, "count_uncertain": uncertain})

    return generation, transform, orbs


def _forward_pool_filter(text: str, match_end: int) -> str:
    """Looks FORWARD from a `CardFactory.Get(Distinct)?ForCombat(`/
    `CardSelectCmd.From*(` call for a `c.Type == CardType.X` or
    `CardPool<X>()` reference within the same statement/expression - these
    appear as the filter argument, which comes AFTER the call itself (unlike
    the loop-count lookbehind used elsewhere)."""
    window = text[match_end : match_end + _FORWARD_LOOKAHEAD_WINDOW]
    type_match = _TYPE_FILTER_RE.search(window)
    if type_match:
        return f"type:{type_match.group(1)}"
    pool_match = _NAMED_POOL_RE.search(window)
    if pool_match:
        return f"pool:{pool_match.group(1)}"
    return "unfiltered"


def _scan_pool_generation(text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for match in _CARD_POOL_GENERATION_RE.finditer(text):
        pool_filter = _forward_pool_filter(text, match.end())
        # A dynamic count feeding this call (e.g. Stoke's `exhaustCount =
        # hand.Count`, computed shortly before the call) can't be read off as
        # a literal - flag rather than assume count=1.
        window_start = max(0, match.start() - _LOOP_LOOKBEHIND_WINDOW)
        uncertain = bool(_COUNT_FROM_COLLECTION_RE.search(text[window_start : match.start()]))
        entries.append({"pool_filter": pool_filter, "count": 1, "count_uncertain": uncertain})
    return entries


def _scan_card_tutor(text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for match in _TUTOR_SELECT_RE.finditer(text):
        pool_filter = _forward_pool_filter(text, match.end())
        entries.append({"pool_filter": pool_filter, "count": 1})
    return entries


def _scan_keywords(text: str) -> list[str]:
    match = _CANONICAL_KEYWORDS_RE.search(text)
    if match is None:
        return []
    body = _extract_balanced_expression(text, match.end())
    return sorted(set(_KEYWORD_RE.findall(body)))


def _scan_synergy_consumers(
    text: str, power_id_by_class: dict[str, str], power_type_by_class: dict[str, str]
) -> tuple[list[dict[str, Any]], int]:
    """Best-effort "payoff" side of a synergy pair: which existing Powers
    does this card's logic read (not apply) to change its own effect, and
    does it consume a channeled orb. See `_POWER_REFERENCE_RE`/
    `_ORB_EVOKE_RE` for what's matched and why."""
    consumers: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for match in _POWER_REFERENCE_RE.finditer(text):
        power_class = match.group("power_class")
        receiver = match.group("receiver")
        referent = "self" if "Owner" in receiver else "enemy"
        key = (power_class, referent)
        if key in seen:
            continue
        seen.add(key)
        power_id = power_id_by_class.get(power_class)
        consumers.append(
            {
                "power_class": power_class,
                "power_id": power_id,
                "power_type": power_type_by_class.get(power_class, "unknown"),
                "referent": referent,
            }
        )
    orb_evoke_count = len(_ORB_EVOKE_RE.findall(text))
    return consumers, orb_evoke_count


def _scan_card_tags(text: str) -> list[str]:
    match = _CANONICAL_TAGS_RE.search(text)
    if match is None:
        return []
    body = _extract_balanced_expression(text, match.end())
    return sorted(set(_CARD_TAG_RE.findall(body)))


def _scan_tribal_references(text: str) -> list[str]:
    """Consumer side of tag-count synergy (Strike/Shiv/... tribal payoffs,
    plus the Soul special case) - see `_TAG_REFERENCE_RE`/
    `_SOUL_TYPE_CHECK_RE`."""
    tags = set(_TAG_REFERENCE_RE.findall(text))
    if _SOUL_TYPE_CHECK_RE.search(text):
        tags.add("Soul")
    return sorted(tags)


def _scan_pile_synergy(text: str) -> dict[str, bool]:
    """Exhaust/discard pile-size synergy - see `_PILE_COUNT_RE`/
    `_EXHAUST_ACTION_RE`/`_DISCARD_ACTION_RE`."""
    piles_referenced = {g1 or g2 for g1, g2 in _PILE_COUNT_RE.findall(text)}
    return {
        "exhaust_pile_reference": "Exhaust" in piles_referenced,
        "discard_pile_reference": "Discard" in piles_referenced,
        "exhausts_other_cards": _EXHAUST_ACTION_RE.search(text) is not None,
        "discards_other_cards": _DISCARD_ACTION_RE.search(text) is not None,
    }


def extract_all(
    rl_root: Path, emulator_root: Path
) -> tuple[list[CardExtraction], dict[str, Any]]:
    cards = _load_json(rl_root / CARDS_JSON_RELATIVE)
    powers = _load_json(rl_root / POWERS_JSON_RELATIVE)
    power_id_by_class = {v["class_name"]: k for k, v in powers.items() if "class_name" in v}
    power_type_by_class = _build_power_type_by_class(emulator_root / POWER_SOURCE_RELATIVE)
    power_stack_type_by_id = {k: v.get("stack_type") for k, v in powers.items()}
    card_id_by_class = {v["class_name"]: k for k, v in cards.items() if "class_name" in v}
    card_source_dir = emulator_root / CARD_SOURCE_RELATIVE

    results: list[CardExtraction] = []
    for card_id, meta in sorted(cards.items()):
        class_name = meta.get("class_name")
        source_file = card_source_dir / f"{class_name}.cs" if class_name else None

        if source_file is None or not source_file.exists():
            no_source_type = meta.get("type")
            excluded, reason = _exclusion_status(card_id, no_source_type, False)
            results.append(
                CardExtraction(
                    card_id=card_id,
                    class_name=class_name or "",
                    energy_cost=meta.get("energy_cost"),
                    rarity=meta.get("rarity"),
                    card_type=no_source_type,
                    target_type=meta.get("target_type"),
                    status="no_source_file",
                    applies_power_state=no_source_type == "Power",
                    deck_eval_excluded=excluded,
                    exclusion_reason=reason,
                )
            )
            continue

        text = source_file.read_text(encoding="utf-8")
        cost, card_type, rarity, target_type = meta.get("energy_cost"), meta.get("type"), meta.get("rarity"), meta.get("target_type")
        ctor_match = _BASE_CTOR_RE.search(text)
        if ctor_match:
            cost = int(ctor_match.group(1))
            card_type, rarity, target_type = ctor_match.group(2), ctor_match.group(3), ctor_match.group(4)
        applies_power_state = card_type == "Power"
        multiplayer_only = _MULTIPLAYER_ONLY_RE.search(text) is not None
        excluded, exclusion_reason = _exclusion_status(card_id, card_type, multiplayer_only)

        # Mechanics living outside CanonicalVars (typically in OnPlay) are
        # scanned over the whole file regardless of whether CanonicalVars
        # itself is present/resolvable below.
        card_generation, card_transform, orbs_channeled = _scan_whole_file_mechanics(text, card_id_by_class)
        pool_generation = _scan_pool_generation(text)
        card_tutor = _scan_card_tutor(text)
        keywords = _scan_keywords(text)
        is_x_cost = _HAS_ENERGY_COST_X_RE.search(text) is not None
        star_cost_match = _STAR_COST_RE.search(text)
        star_cost = int(star_cost_match.group(1)) if star_cost_match else None
        synergy_consumes, orb_evoked = _scan_synergy_consumers(text, power_id_by_class, power_type_by_class)
        synergy_consumes = [c for c in synergy_consumes if c["power_class"] not in _SYNERGY_EXCLUDED_POWER_CLASSES]
        card_tags = _scan_card_tags(text)
        tribal_references = _scan_tribal_references(text)
        pile_synergy = _scan_pile_synergy(text)
        sly_keyword = "Sly" in keywords

        common_fields: dict[str, Any] = dict(
            card_generation=card_generation,
            card_transform=card_transform,
            orbs_channeled=orbs_channeled,
            pool_generation=pool_generation,
            card_tutor=card_tutor,
            keywords=keywords,
            is_x_cost=is_x_cost,
            star_cost=star_cost,
            synergy_consumes=synergy_consumes,
            orb_evoked=orb_evoked,
            card_tags=card_tags,
            tribal_references=tribal_references,
            sly_keyword=sly_keyword,
            **pile_synergy,
            applies_power_state=applies_power_state,
            deck_eval_excluded=excluded,
            exclusion_reason=exclusion_reason,
        )

        vars_match = _CANONICAL_VARS_RE.search(text)
        if vars_match is None:
            results.append(
                CardExtraction(
                    card_id=card_id,
                    class_name=class_name,
                    energy_cost=cost,
                    rarity=rarity,
                    card_type=card_type,
                    target_type=target_type,
                    status="no_canonical_vars",
                    **common_fields,
                )
            )
            continue

        body = _extract_balanced_expression(text, vars_match.end())
        typed_features, powers_applied, opaque_values, recognized_types = _parse_canonical_vars(body)

        for application in powers_applied:
            power_class = application["power_class"]
            power_id = power_id_by_class.get(power_class)
            application["power_id"] = power_id
            application["power_type"] = power_type_by_class.get(power_class, "unknown")
            application["stack_type"] = power_stack_type_by_id.get(power_id) if power_id else None

        matched_count = len(recognized_types)
        broad_count = len(_VAR_CALL_BROAD_RE.findall(body))
        status = "auto"
        unparsed_snippet = None
        if broad_count > matched_count:
            status = "unparsed_residue"
            unparsed_snippet = " ".join(body.split())[:300]

        results.append(
            CardExtraction(
                card_id=card_id,
                class_name=class_name,
                energy_cost=cost,
                rarity=rarity,
                card_type=card_type,
                target_type=target_type,
                status=status,
                features=typed_features,
                powers_applied=powers_applied,
                opaque_values=opaque_values,
                recognized_var_types=sorted(set(recognized_types)),
                unparsed_snippet=unparsed_snippet,
                **common_fields,
            )
        )

    _apply_manual_overrides(results, power_id_by_class, power_type_by_class, power_stack_type_by_id)

    summary = _summarize(results)
    return results, summary


def _exclusion_status(card_id: str, card_type: str | None, multiplayer_only: bool) -> tuple[bool, str]:
    if card_type in _CURSE_STATUS_TYPES or card_id in _CURSE_EQUIVALENT_CARD_IDS:
        return True, "curse_status"
    if multiplayer_only:
        return True, "multiplayer_only"
    return False, ""


def _apply_manual_overrides(
    results: list[CardExtraction],
    power_id_by_class: dict[str, str],
    power_type_by_class: dict[str, str],
    power_stack_type_by_id: dict[str, str | None],
) -> None:
    by_id = {r.card_id: r for r in results}
    for card_id, feature_overrides in _MANUAL_FEATURE_OVERRIDES.items():
        card = by_id.get(card_id)
        if card is not None:
            card.features.update(feature_overrides)
    for card_id, power_overrides in _MANUAL_POWER_OVERRIDES.items():
        card = by_id.get(card_id)
        if card is None:
            continue
        for override in power_overrides:
            power_class = override["power_class"]
            power_id = power_id_by_class.get(power_class)
            card.powers_applied.append(
                {
                    **override,
                    "power_id": power_id,
                    "power_type": power_type_by_class.get(power_class, "unknown"),
                    "stack_type": power_stack_type_by_id.get(power_id) if power_id else None,
                }
            )


def _has_any_signal(r: CardExtraction) -> bool:
    return bool(
        r.features
        or r.powers_applied
        or r.opaque_values
        or r.card_generation
        or r.card_transform
        or r.orbs_channeled
        or r.applies_power_state
        or r.pool_generation
        or r.card_tutor
        or r.is_x_cost
        or r.synergy_consumes
        or r.orb_evoked
        or r.card_tags
        or r.tribal_references
        or r.exhaust_pile_reference
        or r.discard_pile_reference
        or r.exhausts_other_cards
        or r.discards_other_cards
        or r.sly_keyword
    )


def _summarize(results: list[CardExtraction]) -> dict[str, Any]:
    in_scope = [r for r in results if not r.deck_eval_excluded]
    excluded_count = len(results) - len(in_scope)

    status_counts = Counter(r.status for r in results)
    with_typed_features = sum(1 for r in results if r.features)
    with_power_application = sum(1 for r in results if r.powers_applied)
    with_generation = sum(1 for r in results if r.card_generation)
    with_transform = sum(1 for r in results if r.card_transform)
    with_orbs = sum(1 for r in results if r.orbs_channeled)
    with_power_state = sum(1 for r in results if r.applies_power_state)
    with_pool_generation = sum(1 for r in results if r.pool_generation)
    with_tutor = sum(1 for r in results if r.card_tutor)
    with_keywords = sum(1 for r in results if r.keywords)
    with_x_cost = sum(1 for r in results if r.is_x_cost)
    with_star_cost = sum(1 for r in results if r.star_cost is not None)
    with_synergy_consumes = sum(1 for r in results if r.synergy_consumes)
    with_orb_evoked = sum(1 for r in results if r.orb_evoked)
    synergy_consume_tag_counts = Counter(
        f"{c['power_id'] or c['power_class']}:{c['referent']}" for r in results for c in r.synergy_consumes
    )
    with_card_tags = sum(1 for r in results if r.card_tags)
    with_tribal_references = sum(1 for r in results if r.tribal_references)
    card_tag_producer_counts = Counter(tag for r in results for tag in r.card_tags)
    tribal_reference_consumer_counts = Counter(tag for r in results for tag in r.tribal_references)
    with_exhaust_pile_reference = sum(1 for r in results if r.exhaust_pile_reference)
    with_discard_pile_reference = sum(1 for r in results if r.discard_pile_reference)
    with_exhausts_other_cards = sum(1 for r in results if r.exhausts_other_cards)
    with_discards_other_cards = sum(1 for r in results if r.discards_other_cards)
    with_sly_keyword = sum(1 for r in results if r.sly_keyword)
    exclusion_reason_counts = Counter(r.exclusion_reason for r in results if r.exclusion_reason)
    # "Signal"/"residual" are scoped to normal deck-evaluation cards only -
    # Curse/Status cards are out of scope by design (_CURSE_STATUS_TYPES), not
    # a failure of extraction, so they are never counted as residual here.
    with_any_signal = sum(1 for r in in_scope if _has_any_signal(r))
    with_zero_signal = len(in_scope) - with_any_signal
    zero_signal_ids = [r.card_id for r in in_scope if not _has_any_signal(r)]

    debuff_applications = sum(
        1 for r in results for a in r.powers_applied if a["power_type"] == "Debuff"
    )
    buff_applications = sum(
        1 for r in results for a in r.powers_applied if a["power_type"] == "Buff"
    )
    unknown_power_type_applications = sum(
        1 for r in results for a in r.powers_applied if a["power_type"] == "unknown"
    )

    feature_type_counts = Counter()
    for r in results:
        feature_type_counts.update(r.features.keys())

    uncertain_counts = sum(
        1
        for r in results
        for entry in (*r.card_generation, *r.card_transform, *r.orbs_channeled)
        if entry.get("count_uncertain")
    )

    return {
        "total_cards": len(results),
        "deck_eval_excluded_count": excluded_count,
        "in_scope_count": len(in_scope),
        "status_counts": dict(status_counts),
        "cards_with_typed_features": with_typed_features,
        "cards_with_power_application": with_power_application,
        "cards_with_card_generation": with_generation,
        "cards_with_card_transform": with_transform,
        "cards_with_orb_channel": with_orbs,
        "cards_with_power_state_flag": with_power_state,
        "cards_with_pool_generation": with_pool_generation,
        "cards_with_tutor": with_tutor,
        "cards_with_keywords": with_keywords,
        "cards_with_x_cost": with_x_cost,
        "cards_with_star_cost": with_star_cost,
        "cards_with_synergy_consumes": with_synergy_consumes,
        "cards_with_orb_evoked": with_orb_evoked,
        "synergy_consume_tag_counts": dict(synergy_consume_tag_counts),
        "cards_with_card_tags": with_card_tags,
        "cards_with_tribal_references": with_tribal_references,
        "card_tag_producer_counts": dict(card_tag_producer_counts),
        "tribal_reference_consumer_counts": dict(tribal_reference_consumer_counts),
        "cards_with_exhaust_pile_reference": with_exhaust_pile_reference,
        "cards_with_discard_pile_reference": with_discard_pile_reference,
        "cards_with_exhausts_other_cards": with_exhausts_other_cards,
        "cards_with_discards_other_cards": with_discards_other_cards,
        "cards_with_sly_keyword": with_sly_keyword,
        "exclusion_reason_counts": dict(exclusion_reason_counts),
        "cards_with_any_extracted_signal": with_any_signal,
        "cards_with_zero_extracted_signal": with_zero_signal,
        "zero_signal_card_ids": zero_signal_ids,
        "feature_type_coverage": dict(feature_type_counts),
        "power_applications_total": buff_applications + debuff_applications + unknown_power_type_applications,
        "power_applications_buff": buff_applications,
        "power_applications_debuff": debuff_applications,
        "power_applications_unknown_type": unknown_power_type_applications,
        "generation_transform_orb_uncertain_counts": uncertain_counts,
    }


def _print_report(summary: dict[str, Any], results: list[CardExtraction]) -> None:
    total = summary["total_cards"]
    in_scope = summary["in_scope_count"]
    print(f"Total cards in cards.json: {total}")
    print(
        f"Excluded from deck evaluation: {summary['deck_eval_excluded_count']} "
        f"({summary['exclusion_reason_counts']})"
    )
    print(f"In-scope cards: {in_scope}")
    print("Status breakdown (all cards, including excluded):")
    for status, count in sorted(summary["status_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {status:20s} {count:4d}  ({100 * count / total:5.1f}%)")

    signal = summary["cards_with_any_extracted_signal"]
    print(
        f"\nIn-scope cards with at least one extracted signal (typed feature, "
        f"power application, power-state flag, generation/transform/orb, or "
        f"labeled opaque value): {signal}/{in_scope} "
        f"({100 * signal / in_scope:.1f}%)"
    )
    print(f"In-scope cards with zero extracted signal (need full manual authoring): "
          f"{summary['cards_with_zero_extracted_signal']}/{in_scope}")
    print(f"Cards flagged as 'applies a Power/state' (CardType==Power): "
          f"{summary['cards_with_power_state_flag']}")

    print("\nTyped feature coverage (how many cards contribute to each named feature):")
    for feature, count in sorted(summary["feature_type_coverage"].items(), key=lambda kv: -kv[1]):
        print(f"  {feature:15s} {count:4d}")

    print(
        f"\nPower (buff/debuff) applications found via PowerVar<T>: "
        f"{summary['power_applications_total']} "
        f"(buff={summary['power_applications_buff']}, "
        f"debuff={summary['power_applications_debuff']}, "
        f"unknown_type={summary['power_applications_unknown_type']})"
    )
    print(
        f"Cards with card generation: {summary['cards_with_card_generation']}, "
        f"card transform: {summary['cards_with_card_transform']}, "
        f"orb channel: {summary['cards_with_orb_channel']}, "
        f"pool generation: {summary['cards_with_pool_generation']}, "
        f"tutor: {summary['cards_with_tutor']}"
    )
    print(
        f"Cards with keywords: {summary['cards_with_keywords']}, "
        f"X-cost: {summary['cards_with_x_cost']}, "
        f"star cost: {summary['cards_with_star_cost']}"
    )
    print(
        f"\nSynergy 'payoff' side (reads an existing Power, or evokes an "
        f"orb, rather than applying/producing one): "
        f"{summary['cards_with_synergy_consumes']} card(s) reference an "
        f"existing Power, {summary['cards_with_orb_evoked']} card(s) evoke "
        f"an orb"
    )
    for tag, count in sorted(summary["synergy_consume_tag_counts"].items(), key=lambda kv: -kv[1])[:25]:
        print(f"  {tag:30s} {count:3d}")
    print(
        f"\nTribal tag synergy: {summary['cards_with_card_tags']} card(s) declare a "
        f"CanonicalTags producer tag, {summary['cards_with_tribal_references']} "
        f"card(s) count/filter by tag (consumer)"
    )
    print(f"  producer tag counts:  {summary['card_tag_producer_counts']}")
    print(f"  consumer tag counts:  {summary['tribal_reference_consumer_counts']}")
    print(
        f"\nExhaust/discard pile synergy: "
        f"{summary['cards_with_exhaust_pile_reference']} card(s) read Exhaust pile size (consumer), "
        f"{summary['cards_with_exhausts_other_cards']} card(s) exhaust another card (producer); "
        f"{summary['cards_with_discard_pile_reference']} card(s) read Discard pile size (consumer), "
        f"{summary['cards_with_discards_other_cards']} card(s) discard another card (producer); "
        f"{summary['cards_with_sly_keyword']} card(s) have the Sly keyword (auto-plays itself on discard)"
    )
    print(
        f"Generation/transform/orb occurrences with an uncertain (non-literal loop) "
        f"count: {summary['generation_transform_orb_uncertain_counts']}"
    )

    zero_ids = summary["zero_signal_card_ids"]
    print(f"\n{len(zero_ids)} card(s) with ZERO extracted signal across every scanner "
          f"(features, powers, generation, transform, orbs) - true residual needing "
          f"full manual authoring:")
    for card_id in zero_ids:
        print(f"  {card_id}")

    unparsed = [r for r in results if r.status == "unparsed_residue"]
    if unparsed:
        print(f"\n{len(unparsed)} card(s) with unparsed residue (regex gap - inspect these):")
        for r in unparsed[:15]:
            print(f"  {r.card_id}: {r.unparsed_snippet}")

    no_source = [r for r in results if r.status == "no_source_file"]
    if no_source:
        print(f"\n{len(no_source)} card(s) with no matching source file:")
        for r in no_source[:15]:
            print(f"  {r.card_id} (class_name={r.class_name!r})")


_CSV_FEATURE_COLUMNS = (
    "damage",
    "extra_damage",
    "block",
    "cards_drawn",
    "energy",
    "repeat_hits",
    "heal",
    "gold",
    "max_hp",
    "hp_loss",
    "forge",
    "stars",
    "summon",
)
_CSV_COLUMNS = (
    "card_id",
    "class_name",
    "energy_cost",
    "star_cost",
    "is_x_cost",
    "rarity",
    "card_type",
    "target_type",
    "keywords",
    "deck_eval_excluded",
    "exclusion_reason",
    "status",
    "needs_review",
    *_CSV_FEATURE_COLUMNS,
    "applies_power_state",
    "buffs_applied",
    "debuffs_applied",
    "other_powers_applied",
    "card_generation",
    "card_transform",
    "pool_generation",
    "card_tutor",
    "orbs_generated",
    "scales_with_enemy_power",
    "scales_with_self_power",
    "orb_evoked",
    "card_tags",
    "tribal_references",
    "exhaust_pile_reference",
    "discard_pile_reference",
    "exhausts_other_cards",
    "discards_other_cards",
    "sly_keyword",
    "opaque_values",
    "review_snippet",
    "notes",
)


def _format_power_list(powers_applied: list[dict[str, Any]], power_type: str) -> str:
    parts = []
    for p in powers_applied:
        if p["power_type"] != power_type:
            continue
        label = p["power_id"] or p["power_class"]
        suffix = f"({p['stack_type']})" if p.get("stack_type") else ""
        parts.append(f"{label}:{_format_number(p['base_value'])}{suffix}")
    return ";".join(parts)


def _format_number(value: float) -> str:
    return f"{value:g}"


def _format_created_card_list(entries: list[dict[str, Any]]) -> str:
    parts = []
    for entry in entries:
        label = entry["created_card_id"] or entry["created_class"]
        count = entry["count"]
        suffix = "?" if entry.get("count_uncertain") else ""
        parts.append(f"{label}:{count}{suffix}")
    return ";".join(parts)


def _format_orb_list(entries: list[dict[str, Any]]) -> str:
    parts = []
    for entry in entries:
        count = entry["count"]
        suffix = "?" if entry.get("count_uncertain") else ""
        parts.append(f"{entry['orb_class']}:{count}{suffix}")
    return ";".join(parts)


def _format_pool_list(entries: list[dict[str, Any]]) -> str:
    parts = []
    for entry in entries:
        count = entry["count"]
        suffix = "?" if entry.get("count_uncertain") else ""
        parts.append(f"{entry['pool_filter']}:{count}{suffix}")
    return ";".join(parts)


def _format_synergy_consumes(entries: list[dict[str, Any]], referent: str) -> str:
    parts = []
    for entry in entries:
        if entry["referent"] != referent:
            continue
        label = entry["power_id"] or entry["power_class"]
        parts.append(label)
    return ";".join(parts)


def _needs_review(card: CardExtraction) -> bool:
    if card.deck_eval_excluded:
        return False
    if not _has_any_signal(card):
        return True
    if card.status == "unparsed_residue":
        return True
    return any(
        entry.get("count_uncertain")
        for entry in (
            *card.card_generation,
            *card.card_transform,
            *card.orbs_channeled,
            *card.pool_generation,
        )
    )


def _card_to_csv_row(card: CardExtraction) -> dict[str, str]:
    row: dict[str, str] = {
        "card_id": card.card_id,
        "class_name": card.class_name,
        "energy_cost": "" if card.energy_cost is None else str(card.energy_cost),
        "star_cost": "" if card.star_cost is None else str(card.star_cost),
        "is_x_cost": "TRUE" if card.is_x_cost else "",
        "rarity": card.rarity or "",
        "card_type": card.card_type or "",
        "target_type": card.target_type or "",
        "keywords": ";".join(card.keywords),
        "deck_eval_excluded": "TRUE" if card.deck_eval_excluded else "",
        "exclusion_reason": card.exclusion_reason,
        "status": card.status,
        "needs_review": "TRUE" if _needs_review(card) else "",
        "applies_power_state": "TRUE" if card.applies_power_state else "",
        "buffs_applied": _format_power_list(card.powers_applied, "Buff"),
        "debuffs_applied": _format_power_list(card.powers_applied, "Debuff"),
        "other_powers_applied": ";".join(
            f"{p['power_id'] or p['power_class']}:{_format_number(p['base_value'])}"
            for p in card.powers_applied
            if p["power_type"] not in ("Buff", "Debuff")
        ),
        "card_generation": _format_created_card_list(card.card_generation),
        "card_transform": _format_created_card_list(card.card_transform),
        "pool_generation": _format_pool_list(card.pool_generation),
        "card_tutor": _format_pool_list(card.card_tutor),
        "orbs_generated": _format_orb_list(card.orbs_channeled),
        "scales_with_enemy_power": _format_synergy_consumes(card.synergy_consumes, "enemy"),
        "scales_with_self_power": _format_synergy_consumes(card.synergy_consumes, "self"),
        "orb_evoked": str(card.orb_evoked) if card.orb_evoked else "",
        "card_tags": ";".join(card.card_tags),
        "tribal_references": ";".join(card.tribal_references),
        "exhaust_pile_reference": "TRUE" if card.exhaust_pile_reference else "",
        "discard_pile_reference": "TRUE" if card.discard_pile_reference else "",
        "exhausts_other_cards": "TRUE" if card.exhausts_other_cards else "",
        "discards_other_cards": "TRUE" if card.discards_other_cards else "",
        "sly_keyword": "TRUE" if card.sly_keyword else "",
        "opaque_values": ";".join(f"{k}:{_format_number(v)}" for k, v in card.opaque_values.items()),
        "review_snippet": card.unparsed_snippet or "",
        "notes": "",
    }
    for column in _CSV_FEATURE_COLUMNS:
        value = card.features.get(column)
        row[column] = "" if value is None else _format_number(value)
    return row


def _write_csv(results: list[CardExtraction], path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists - refusing to overwrite a possibly hand-edited file. "
            "Pass --force to regenerate it from scratch (this discards any manual edits)."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for card in results:
            writer.writerow(_card_to_csv_row(card))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rl-root", type=Path, default=DEFAULT_RL_ROOT)
    parser.add_argument("--emulator-root", type=Path, default=DEFAULT_EMULATOR_ROOT)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("tools/output/card_secondary_features.csv"),
        help="human-editable output CSV (one row per card); refuses to overwrite an existing file unless --force is given",
    )
    parser.add_argument("--force", action="store_true", help="allow --csv to overwrite an existing file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional path to (always) write the full raw per-card JSON dump - disposable, not meant for hand-editing",
    )
    args = parser.parse_args()

    results, summary = extract_all(args.rl_root, args.emulator_root)
    _print_report(summary, results)

    try:
        _write_csv(results, args.csv, force=args.force)
    except FileExistsError as exc:
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(1)
    print(f"\nWrote human-editable card feature table to {args.csv} ({len(results)} rows)")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "cards": [asdict(r) for r in results]}
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"Wrote full per-card extraction (raw, disposable) to {args.output}")


if __name__ == "__main__":
    main()
