# CombatDecisionEngine の使い方

`sts2_training.decision` は `AsyncTrainingApiClient` (DTO v0.7) の上で、
Policy で候補を絞り、Beam Search で `emulate_actions` し、Value で盤面を評価して
root の action を選ぶための薄い意思決定レイヤーです。

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
    time_budget_ms=200.0,       # 指定する場合は正数。
    expand_partial=True,
    release_branches_on_finish=True,
)

engine = CombatDecisionEngine(client, beam_config=config)
```

`beam_searchable_action_types` の既定値は `system` / `card` / `potion` です。
この判定は root だけでなく各 depth の次 decision に対しても行われます。
探索途中で reward / map / event など非 Combat の境界へ移った node は、それ以上
branch せず、その時点の Value を持つ finished node として扱います。

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

## ValueModel

`ValueModel` は各 `masked_emulator_dto` を「大きいほど良い」スカラーへ変換します。
学習済みモデルでは `evaluate_batch()` を直接 override してください。

`BeamSearchEngine` は `evaluate_batch()` の戻り件数が入力 DTO 件数と一致することを
検証します。不一致は候補を黙って捨てず `RuntimeError` として表面化します。

既定の `HeuristicValueFunction` は HP、block、敵 HP、予測被弾、Power などの簡易特徴を
使います。block は複数敵の攻撃合計に対して一度だけ差し引かれます。

## フォールバックと例外

次のケースでは Beam Search の結果に action が無いため
`HeuristicCombatSelector` へフォールバックします。

- root が非 Combat 境界
- `emulate_actions` batch が RL に reject された
- 制限時間内に探索候補を得られなかった

一方、custom `PolicyModel` / `ValueModel` の予期しない例外は**握りつぶしません**。
実装バグを「正常な heuristic decision」に変換すると、壊れた model deployment を
検知できなくなるためです。Transport / protocol の例外も通常どおり caller へ伝播します。

## timeout

`decide()` は `get_decision` 後の残り時間だけを Beam Search に渡します。
`decide_and_commit()` はさらに同じ全体 budget の残り時間だけを `commit_action` に渡します。
以前のように残り時間が無くても 50ms を水増しして API を呼ぶことはありません。

## Branch cleanup

探索で作った Branch は `finally` で cleanup します。

1. `cancel_branches`
2. client が引き続き fresh request を送れる状態なら `release_branches`

cancel が失敗しても session が有効なら release は別途試します。ただし
`pending_retry` または `session_invalid` が立った場合は、新しい cleanup request を
送ると session sequencing を壊すため中止します。

## 現在の既定実装について

`PriorHeuristicPolicy` と `HeuristicValueFunction` は学習済みモデルではなく、
モデル接続前でも pipeline を end-to-end で動かすための簡易実装です。
本番の意思決定品質を担保するものではありません。
