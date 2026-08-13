# selection モジュール

## 0. 文章の目的

この文書は `src/sts2_training/selection/` の非 Combat / fallback heuristic selection を説明する。Combat Beam Search の詳細は [02_decision_core.md](02_decision_core.md) に分け、ここでは legal action の分類、reward card、map room、event option、canonical choice-card semantics を扱う。

## 1. 概要

`selection/` は初期 self-play や fallback で使う action selector 群である。`HeuristicCombatSelector` は available legal actions を action type ごとに分類し、choice-card、card、map room、event option などに専用 scoring があればそれを使い、最後は候補から選ぶ。

Training は canonical `pendingChoice.choiceSemantics` を消費し、prompt text、label、card id、option shape から mechanics を推測しない。未知または future semantics は `operation="unknown"` に degrade し、policy-neutral に扱う。

## 2. Architecture

| ファイル | 役割 |
|---|---|
| `action_classification.py` | `legal_actions` を action_type で filter/group する |
| `choice_semantics.py` | `ChoiceCardSemantics` と `PendingChoiceContext` の defensive parser |
| `choice_card_heuristic.py` | canonical choice-card の option preference |
| `reward_card_selection.py` | reward card selection policy。random と card-data score based |
| `room_heuristic.py` | map room の metadata-only preference |
| `event_choice_heuristic.py` | confirmed lethal event option を除外する safety filter |
| `heuristic_selector.py` | 上記を統合する `HeuristicCombatSelector` |

`action_classification.available_actions()` は `is_available is not False` の action だけを残す。つまり `is_available` が明示的に `False` の action だけを除外し、field がない action や `None`/`0`/空文字など他の falsy value は除外しない。各 type helper は order を保って filter し、`group_by_action_type()` は available action を `action_type` ごとの dict にする。

`RewardCardSelectionPolicy` は Protocol で、`RandomRewardCardSelectionPolicy` と `CardDataRewardCardSelectionPolicy` がある。後者は sts2log.com card stats export 由来の `skada_score` を参照する。

## 3. API

```python
available_actions(legal_actions) -> list[Mapping[str, Any]]
card_actions(legal_actions) -> list[Mapping[str, Any]]
choice_card_actions(legal_actions) -> list[Mapping[str, Any]]
reward_card_actions(legal_actions) -> list[Mapping[str, Any]]
map_room_actions(legal_actions) -> list[Mapping[str, Any]]
choice_event_option_actions(legal_actions) -> list[Mapping[str, Any]]
group_by_action_type(legal_actions) -> dict[str, list[Mapping[str, Any]]]
```

```python
@dataclass(frozen=True)
class ChoiceCardSemantics:
    version: int
    operation: str
    ...
    def is_known(self) -> bool

pending_choice_context(masked_emulator_dto) -> PendingChoiceContext | None
choice_option_id(action) -> str | None
```

```python
class HeuristicCombatSelector:
    def select(self, legal_actions: Sequence[Mapping[str, Any]], masked_emulator_dto: Mapping[str, Any] | None = None) -> dict[str, Any]

class NoAvailableActionError(RuntimeError):
    ...
```

## 4. 使用例

```python
from sts2_training.selection import HeuristicCombatSelector

masked_emulator_dto = {
    "legal_actions": [
        {"action_id": "a", "action_type": "choice_skip", "is_available": True},
        {"action_id": "b", "action_type": "choice_reward_card", "is_available": False},
    ]
}
legal_actions = masked_emulator_dto["legal_actions"]

selector = HeuristicCombatSelector()
action = selector.select(legal_actions, masked_emulator_dto)
assert action["action_id"] == "a"
```

choice semantics を直接読む例:

```python
from sts2_training.selection import choice_option_id, pending_choice_context

context = pending_choice_context(masked_emulator_dto)
option_id = choice_option_id({
    "action_type": "choice_card",
    "parameters": {"optionId": "opt-1"},
})
```

## 5. 補足説明

`event_choice_heuristic.safe_event_option_candidates()` は `willKillPlayer is True` が明示された option だけを除外する。未知の lethal risk を推測で除外しない。canonical card choice semantics の背景は古い [choice_card_semantics.md](choice_card_semantics.md) にも残っているが、この文書では current parser の挙動を優先する。
