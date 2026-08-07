# HeuristicCombatSelector の使い方

初期段階（簡単なロジックによる選択＋ログ収集）向けの、最小構成の選択器です。
`action_type` によるカテゴリ分けと優先順位ディスパッチだけを行い、カテゴリ内はランダム選択します。

## 必要なデータ

`HeuristicCombatSelector.select()` が要求するのは、`get_decision` / `commit_action` /
`emulate_action` のレスポンスに含まれる以下の1点だけです。

```
response["masked_emulator_dto"]["legal_actions"]
```

`legal_actions` は辞書のリストで、各要素に以下のキーが必要です（`rl_training_dto_documentation.md`
の「そのまま公開する情報 / Legal Actions」に対応）。

| キー | 型 | 必須 | 用途 |
|---|---|---|---|
| `action_id` | string | Yes | 選ばれたら `commit_action`/`emulate_action` にそのまま渡す |
| `action_type` | string | Yes | カテゴリ分け・優先順位判定に使用 |
| `is_available` | bool | No（省略時は利用可能扱い） | `False` の場合は選択対象から除外 |
| `parameters` | object | No | 選択器自体は参照しない（呼び出し側が必要なら利用） |

これ以外のフィールド（`label` など）は無視します。`resolved.normalizedChoiceOperation` の
ような OLD プロジェクト（`C:\STS2_Training_OLD`）の教師データ由来のフィールドは、現行の
ライブ API レスポンスには存在しないため、この選択器は一切参照しません。

## 使い方

```python
import random

from sts2_training.api.client import TrainingApiClient
from sts2_training.selection import HeuristicCombatSelector

selector = HeuristicCombatSelector(rng=random.Random(seed))

decision = api_client.get_decision(instance_id, "root", timeout_s=120.0)
legal_actions = decision["masked_emulator_dto"]["legal_actions"]

chosen = selector.select(legal_actions)

result = api_client.commit_action(
    instance_id,
    decision["decision_point_id"],
    chosen["action_id"],
    timeout_s=120.0,
)
```

`legal_actions` が空、または全て `is_available=False` の場合は
`sts2_training.selection.NoAvailableActionError` を送出します。呼び出し側でこの例外を
捕捉し、instance の終了処理（`close_instance`）に進んでください。

## 選択ロジックの中身

1. `legal_actions` から `is_available is not False` のものだけを残す。
2. `action_type` ごとにグルーピングする。
3. 次の優先順位で最初に候補が存在するカテゴリを採用する。
   `card` → `choice_card` → `choice_confirm` → `choice_skip`
4. 上記4種のいずれにも該当するアクションが無い場合（`potion` や `end_turn` など）は、
   利用可能な全アクションから選ぶ。
5. 採用したカテゴリ内では `random.Random` で一様ランダムに1つ選ぶ。

## 既知の制約・今後の拡張ポイント

- 現状は完全にランダムな選択であり、カード効果やゲーム状態を一切見ていません。
  「簡単なロジック」段階の土台であり、実質的な意思決定ロジックはまだ入っていません。
- カテゴリごとの選択処理は `HeuristicCombatSelector._choose()` に集約されています。
  将来「`card` カテゴリだけ攻撃力ヒューリスティックを入れる」「`choice_card` だけ
  学習済みモデルに差し替える」といった拡張は、他カテゴリに影響を与えずに
  `_choose` をカテゴリ単位で分岐させることで対応できます。
- `card` と `choice_card` が同一 decision の `legal_actions` に混在するケースがあるかは
  未確認です。混在しない前提が実機で確認できれば、優先順位ロジックは単純化できます。
- 選択理由やカテゴリ情報をログに残す仕組み（`decision_id` 採番、`action_type` の記録など）
  はこのモジュールの範囲外です。別途ログ収集レイヤー側で実装してください。
