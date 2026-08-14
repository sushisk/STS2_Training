# board_eval モジュール

## 0. 文章の目的

この文書は `src/sts2_training/board_eval/` の Run-state / deck evaluation 実装を説明する。Combat 中の immediate value ではなく、Run 全体の win probability 推定に使うカード特徴、deck summary、DTO adapter、training data、linear inference interface が対象である。

## 1. 概要

`board_eval/` は masked emulator DTO から deck と board context を抽出し、Run-level value model に渡せる特徴へ変換する。カード特徴は CSV から読み込まれ、upgrade 有無を反映した `CardFeatures` として表現される。deck 全体は `DeckSummary` に集約され、HP/floor/gold などの context と合わせて教師データになる。

この層は Combat search の `ValueModel` とは別である。`RunStateValueModel.predict_win_probability()` は `Mapping[str, object]` を受け取って 0.0 から 1.0 の確率を返す interface で、`LinearRunStateValueModel` は JSON artifact から標準ライブラリのみで推論する baseline 実装である。

## 2. Architecture

| ファイル | 役割 |
|---|---|
| `card_features.py` | static CSV から `CardFeatures` を作る。dynamic scaling は `ReferenceScaling` として保持する |
| `deck_summary.py` | card features の sequence から deck-size、type count、effect total/per-energy などを集約する |
| `dto_adapter.py` | masked emulator DTO の deck/board fields を `DeckCardRef`、features、summary、context に変換する |
| `run_state_value.py` | `RunStateValueModel` abstract base |
| `linear_value_function.py` | linear logits + sigmoid による dependency-free inference |
| `training_data.py` | self-play JSONL selection logs から `BoardStateExample` を作り、CSV/JSONL 行に落とす |

`CardFeatureExtractor` は card id と upgraded flag を分けて扱う。未知カードは `UnknownCardError` で fail でき、DTO adapter では `UnknownCardPolicy = "skip" | "raise"` を選べる。

dynamic reference は `ScalingSource` と `Scope` で表現される。固定 damage/block へ無理に畳み込まず、combat/run/turn 依存の情報として `dynamic_scalings` に残す設計である。

## 3. API

```python
class CardFeatureExtractor:
    @classmethod
    def from_csv(cls, path: Path | str = DEFAULT_CARD_FEATURES_CSV) -> "CardFeatureExtractor"
    def card_ids(self) -> list[str]
    def extract(self, card_id: str, *, upgraded: bool = False) -> CardFeatures
    def extract_all(self, *, upgraded_ids: Iterable[str] = ()) -> dict[str, CardFeatures]

@dataclass(frozen=True)
class CardFeatures:
    def uses_star_cost(self) -> bool
    def is_dynamic(self) -> bool
    def to_vector(self) -> tuple[float, ...]
```

```python
summarize_deck(cards: Sequence[CardFeatures]) -> DeckSummary

@dataclass(frozen=True)
class DeckSummary:
    def to_vector(self) -> tuple[float, ...]
```

```python
deck_card_refs_from_dto(dto) -> list[DeckCardRef]
deck_features_with_unknown_count_from_dto(dto, extractor, *, on_unknown_card="raise") -> tuple[list[CardFeatures], int]
deck_features_from_dto(dto, extractor, *, on_unknown_card="raise") -> list[CardFeatures]
deck_summary_from_dto(dto, extractor, *, on_unknown_card="raise") -> DeckSummary
state_kind_from_dto(dto) -> str | None
board_context_from_dto(dto) -> dict[str, object]
```

```python
class RunStateValueModel(ABC):
    def predict_win_probability(self, run_state: Mapping[str, object]) -> float
```

## 4. 使用例

```python
from sts2_training.board_eval import (
    CardFeatureExtractor,
    board_context_from_dto,
    deck_summary_from_dto,
)

dto = {
    "deck": [{"card_id": "Strike_R", "upgraded": False}],
    "floor": 3,
    "player": {"hp": 60, "max_hp": 70},
    "gold": 99,
}

extractor = CardFeatureExtractor.from_csv()
summary = deck_summary_from_dto(dto, extractor, on_unknown_card="skip")
context = board_context_from_dto(dto)
features = {
    **context,
    "deck_size": summary.deck_size,
    "attack_count": summary.attack_count,
}
```

教師データ生成 CLI:

```bash
python -m sts2_training.board_eval.training_data \
  --log-dir data/self_play \
  --output data/board_eval_examples.jsonl \
  --on-unknown-card skip
```

## 5. 補足説明

旧 `sts2_deck_evaluation_rl_implementation.md`(削除済み)は将来案としての neural/embedding 構想を多く含んでいたが、現在の source は feature extraction、summary、linear value interface、log example builder までが実装範囲である。将来の拡張構想は本文書のスコープではなく、現在動く実装だけを記載する。非 Combat selector から board context を使う場合は [06_selection.md](06_selection.md) を参照する。
