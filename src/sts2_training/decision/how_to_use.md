# CombatDecisionEngine の使い方

`sts2_training.decision` は `AsyncTrainingApiClient` (DTO v0.7) の上で、
Policy で候補を絞り、Beam Search で `emulate_actions` し、Value で盤面を評価して
root の action を選ぶための薄い意思決定レイヤーです。
`CombatDecisionEngine` は root 専用です。非 root Branch の decision を
`commit_action` する API ではありません。

## 基本フロー

```text
get_decision(root)
    -> PolicyModel.propose_batch()
    -> emulate_actions([...])
    -> ValueModel.evaluate_batch()
    -> beam prune / next depth
    -> commit_action(root action)
```

1 depth は 1 つの**論理 batch**として扱います。ただし実際の API request は、
`BeamSearchConfig.max_batch_size` と RL が `start_instance` で公開した
`max_emulate_actions_items` の小さい方を上限として複数 request に分割されます。

## 最小構成

```python
from sts2_training.api import AsyncTrainingApiClient, TcpConnection
from sts2_training.decision import CombatDecisionEngine

connection = TcpConnection(host="127.0.0.1", port=8765)
async with AsyncTrainingApiClient(connection) as client:
    instance_id = await client.start_instance(
        {"instance_type": "combat"}, timeout_s=30.0
    )

    # client / instance の寿命に対して 1 個を使い回す。
    engine = CombatDecisionEngine(client)

    response = await engine.decide_and_commit(
        instance_id,
        timeout_s=2.0,
    )
```

`CombatDecisionEngine` を使い回すのは、内部の `BeamSearchEngine` が Branch ID / RNG ID
の allocator を保持し、探索をまたいで ID を再利用しないためです。

## 設定

```python
from sts2_training.decision import BeamSearchConfig, CombatDecisionEngine

config = BeamSearchConfig(
    beam_width=8,
    top_k_actions=4,
    max_depth=2,
    max_batch_size=64,          # ローカル上限。RL capability が小さければそちらを優先。
    time_budget_ms=200.0,       # 指定する場合は有限の正数。
    expand_partial=True,
    release_branches_on_finish=True,
)

engine = CombatDecisionEngine(client, beam_config=config)
```

`beam_searchable_action_types` の既定値は `system` / `card` / `potion` です。
この判定は root だけでなく各 depth の次 decision に対しても行われます。
明示的に `is_available=False` の action は探索可能性の判定から除外します。
探索途中で reward / map / event など非 Combat の境界へ移った node は、それ以上
branch せず、その時点の Value を持つ finished node として扱います。

## 探索モード(`search_modes.py`)

`BeamSearchConfig`を毎回手組みしなくても済むよう、いくつかの名前付きプリセットを
`SEARCH_MODES`として用意しています。

| モード名 | max_depth | beam_width | top_k_actions | 用途 |
|---|---|---|---|---|
| `shallow` | 1 | 4 | 3 | 先読み最小・データ収集のスループット優先 |
| `standard`(既定) | 2 | 8 | 4 | `BeamSearchConfig`のデフォルトと同じ |
| `deep` | 4 | 8 | 4 | 幅は変えずに先読みを深く(評価・デバッグ向け) |
| `wide` | 2 | 16 | 6 | 深さは変えずbeam幅/候補数を広く |

`resolve_search_mode(mode, *, max_depth=None)`が変換の窓口です。`mode`には
モード名(str)/`BeamSearchConfig`インスタンス(手組み)/`None`(既定モード)の
いずれも渡せます。`max_depth`を指定すると、選んだモードの`max_depth`だけを
上書きします(`dataclasses.replace`、他のフィールドはプリセットのまま) -
**これが「ビームサーチの深さをトップレベルから変更する」ための直接の窓口**です。

```python
from sts2_training.decision import resolve_search_mode, CombatDecisionEngine

# モード名だけ
engine = CombatDecisionEngine(client, beam_config=resolve_search_mode("deep"))

# モードを選びつつ深さだけ上書き
engine = CombatDecisionEngine(client, beam_config=resolve_search_mode("wide", max_depth=5))
```

`runner`パッケージの2つの入り口(`start_combat_from_state` / `start_new_run`)は、`engine`を渡す
代わりに`search_mode`/`beam_max_depth`をそのまま受け取れます(内部で
`resolve_search_mode`を呼びます) - 詳細は`runner/how_to_use.md`。

## PolicyModel

`PolicyModel` は action を best-first で提案します。Beam Search は
`propose_batch()` の戻り順と `top_k` を使って探索候補を決めます。

```python
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.decision import ActionCandidate, PolicyModel


class MyPolicy(PolicyModel):
    def propose(
        self,
        legal_actions: Sequence[Mapping[str, Any]],
        masked_emulator_dto: Mapping[str, Any],
        *,
        top_k: int,
    ) -> list[ActionCandidate]:
        ranked = sorted(legal_actions, key=my_score, reverse=True)
        return [ActionCandidate(action_id=a["action_id"]) for a in ranked[:top_k]]
```

学習済みモデルでは `propose_batch()` を直接 override して一括推論してください。
`ActionCandidate` は探索に必要な `action_id` だけを持ちます。使われない prior や
action type の複製は持たせません。

`BeamSearchEngine` は `propose_batch()` の batch cardinality、各 node の候補数、
`ActionCandidate` 型、および `action_id` が現在 available な legal action であることを
検証します。不正な model 出力は RL の batch reject に変換せず、`RuntimeError` として
早期に表面化します。

## ValueModel

`ValueModel` は各 `masked_emulator_dto` を「大きいほど良い」スカラーへ変換します。
学習済みモデルでは `evaluate_batch()` を直接 override してください。

`BeamSearchEngine` は `evaluate_batch()` の戻り件数が入力 DTO 件数と一致することに加え、
各値が有限な数値であることも検証します。不一致、`NaN`、`inf`、非数値は候補を黙って
捨てず `RuntimeError` として表面化します。root 自体は最終候補にならないため、探索開始時の
余計な singleton `evaluate()` 呼び出しは行いません。

既定の `HeuristicValueFunction` は HP、block、敵 HP、予測被弾、Power などの簡易特徴を
使います。block は複数敵の攻撃合計に対して一度だけ差し引かれます。敵 HP 比率の分母には
倒した敵の max HP も残すため、敵を倒した結果だけで進捗評価が逆向きになることを避けます。

## フォールバックと例外

次のケースでは Beam Search の結果に action が無いため
`HeuristicCombatSelector` へフォールバックします。

- root が非 Combat 境界
- `emulate_actions` batch が RL に reject され、かつ client session が引き続き fresh な場合
- 制限時間内に探索候補を得られなかった

一方、custom `PolicyModel` / `ValueModel` の予期しない例外や不正出力は**握りつぶしません**。
`ApiProtocolError` は completion-uncertain で exact replay が必要になるためフォールバックせず、
`faulted` operation や session を invalid にした reject も caller へ伝播します。
壊れた model / protocol / session を「正常な heuristic decision」に変換しないことを優先します。

## timeout

`decide()` は `get_decision` 後の残り時間だけを Beam Search に渡します。
`decide_and_commit()` はさらに同じ全体 budget の残り時間だけを `commit_action` に渡します。
残り時間が無いのに固定の猶予を水増しして API を呼ぶことはありません。

Beam Search の branch cleanup も search に渡された全体 timeout の残り時間内だけで行います。
cleanup 1 request の上限は 5 秒ですが、残り budget の方が短ければそちらを使います。

## Branch cleanup

探索で作った Branch は `finally` で cleanup します。

1. `cancel_branches`
2. client が引き続き fresh request を送れる状態なら `release_branches`

既知の recoverable な reject、または completion が確実に uncertain ではない transport failure は
best-effort cleanup としてログに残して続行します。一方、protocol error、faulted operation、
completion-uncertain transport error、予期しない実装例外は、正常な探索結果を返すために
握りつぶしません。すでに search/model 側の例外を伝播中なら、その元の例外を優先し、cleanup の
二次障害はログに残します。

`pending_retry` または `session_invalid` が立った場合は、新しい cleanup request を送ると
session sequencing を壊すため中止します。

## 現在の既定実装について

`PriorHeuristicPolicy` と `HeuristicValueFunction` は学習済みモデルではなく、
モデル接続前でも pipeline を end-to-end で動かすための簡易実装です。
本番の意思決定品質を担保するものではありません。
