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
    -> choice_* continuation は local macro-action として解決
    -> stable / terminal state だけ ValueModel.evaluate_batch()
    -> beam prune / next combat depth
    -> commit_action(root action)
```

通常の card / potion / system action が `max_depth` を消費します。
`choice_target` / `choice_card` / `choice_confirm` / `choice_skip` は continuation として
`max_continuation_steps` の範囲で解決され、未解決 continuation DTO は通常の
`ValueModel` の入力や最終 actionable candidate にはなりません。

1 combat depth は 1 つの**論理 batch**として扱います。ただし実際の API request は、
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

`CombatDecisionEngine(client)` の既定 semantic scope は full Combat domain です。
つまり `system` / `card` / `potion` に加えて Combat 中の interactive continuation も
Beam Search の対象になります。

`CombatDecisionEngine` を使い回すのは、内部の `BeamSearchEngine` が Branch ID / RNG ID
の allocator を保持し、探索をまたいで ID を再利用しないためです。

## 設定

```python
from sts2_training.decision import BeamSearchConfig, CombatDecisionEngine

config = BeamSearchConfig(
    beam_width=8,
    top_k_actions=4,
    max_depth=2,
    max_continuation_steps=8,
    max_batch_size=64,          # ローカル上限。RL capability が小さければそちらを優先。
    time_budget_ms=200.0,       # 指定する場合は有限の正数。
    expand_partial=True,
    release_branches_on_finish=True,
)

engine = CombatDecisionEngine(client, beam_config=config)
```

### Beam semantic scope の優先順位

`BeamSearchConfig.beam_searchable_action_types` は引き続き公開設定です。
**明示的な `beam_config` を `CombatDecisionEngine` に渡した場合、その
`beam_searchable_action_types` はそのまま保持され、wrapper が黙って拡張しません。**

例えば continuation を対象外にした legacy / narrow scope は次のように指定できます。

```python
legacy_scope = frozenset({"system", "card", "potion"})
config = BeamSearchConfig(
    beam_searchable_action_types=legacy_scope,
)
engine = CombatDecisionEngine(client, beam_config=config)
```

この場合、`choice_*` decision は Beam Search されず、通常の fallback 経路へ進みます。

`CombatDecisionEngine` には semantic scope を明示する `beam_action_types` もあります。
優先順位は次のとおりです。

1. `beam_config` を渡さず `beam_action_types` も省略: full Combat scope
2. `beam_config` だけを渡す: `beam_config.beam_searchable_action_types` を保持
3. `beam_action_types` だけを渡す: その scope を使用
4. 両方を渡す: 2つの scope が一致する場合のみ許可。異なる場合は `ValueError`

したがって、同じ `BeamSearchConfig` を低レベル `BeamSearchEngine` に直接渡す場合と
`CombatDecisionEngine` 経由で渡す場合で、その明示 semantic scope が黙って変わることはありません。

`beam_searchable_action_types` の判定は root だけでなく各 depth の次 decision に対しても
行われます。明示的に `is_available=False` の action は探索可能性の判定から除外します。
探索途中で reward / map / event など非 Combat の境界へ移った node は、それ以上
branch しません。

## 探索モード (`search_modes.py`)

`BeamSearchConfig` を毎回手組みしなくても済むよう、いくつかの名前付きプリセットを
`SEARCH_MODES` として用意しています。

| モード名 | max_depth | beam_width | top_k_actions | 用途 |
|---|---|---|---|---|
| `shallow` | 1 | 4 | 3 | 先読み最小・データ収集のスループット優先 |
| `standard` (既定) | 2 | 8 | 4 | 標準の latency-quality tradeoff |
| `deep` | 4 | 8 | 4 | 幅は変えずに先読みを深く |
| `wide` | 2 | 16 | 6 | 深さは変えず beam 幅 / 候補数を広く |

named mode 自体は depth / width / top-k などの performance budget を表します。
`resolve_search_mode(mode, *, max_depth=None)` が変換の窓口です。`mode` には
モード名 (str) / `BeamSearchConfig` インスタンス / `None` のいずれも渡せます。
`max_depth` を指定すると、選んだモードの `max_depth` だけを上書きします。

```python
from sts2_training.decision import resolve_search_mode, CombatDecisionEngine

engine = CombatDecisionEngine(client, beam_config=resolve_search_mode("deep"))
engine = CombatDecisionEngine(
    client,
    beam_config=resolve_search_mode("wide", max_depth=5),
)
```

注意: `resolve_search_mode()` の戻り値は `BeamSearchConfig` です。その config を
`CombatDecisionEngine` へ**明示的に渡した場合は、その config に入っている
`beam_searchable_action_types` も明示設定として保持されます**。full Combat scope を
使いたい場合は、config 側の scope も full Combat scope にするか、config と一致する
`beam_action_types` を指定してください。矛盾する二重指定は拒否されます。

`runner` パッケージの2つの入り口 (`start_combat_from_state` / `start_new_run`) は、
`engine` を渡す代わりに `search_mode` / `beam_max_depth` をそのまま受け取れます
(内部で `resolve_search_mode` を呼びます)。詳細は `runner/how_to_use.md` を参照してください。

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
`ActionCandidate` は探索に必要な `action_id` だけを持ちます。

`CombatDecisionEngine` は supplied `PolicyModel` を `CoverageConstrainedPolicy` で包みます。
このため learned / batch-only Policy に置き換えても、End Turn / card / potion / continuation
completion などの structural branch recall を Policy ranking 自体へ混ぜずに維持できます。

`BeamSearchEngine` は `propose_batch()` の batch cardinality、各 node の候補数、
`ActionCandidate` 型、および `action_id` が現在 available な legal action であることを
検証します。不正な model 出力は `RuntimeError` として早期に表面化します。

## ValueModel

`ValueModel` は resolved stable / terminal `masked_emulator_dto` を「大きいほど良い」
スカラーへ変換します。学習済みモデルでは `evaluate_batch()` を直接 override してください。

pending continuation state は global `ValueModel` の domain ではありません。
continuation は local macro-action として Policy order / local beam width で解決し、stable または
terminal state に戻った後で Value 評価されます。

`BeamSearchEngine` は `evaluate_batch()` の戻り件数が入力 DTO 件数と一致することに加え、
各値が有限な数値であることも検証します。不一致、`NaN`、`inf`、非数値は候補を黙って
捨てず `RuntimeError` として表面化します。

既定の `HeuristicValueFunction` は、プレイヤーの生存を
`effective_hp = hp − max(0, 敵の攻撃予告合計 − block)` の 1 項（`effective_hp_ratio`）で表し、
これに敵 HP、生存敵数、Power の特徴を足します。

HP と block を別項にしない理由は、Beam がターン内の異なる時点にある leaf を比較するためです。
「カードを打ったがまだ敵のターンを受けていない」leaf と「ターンを終えて既に受けた」leaf を
素の HP / block で比べると、未払いの被弾量が揃わず、プレイの良し悪しではなくターンの偶奇で
勝敗が決まります。`effective_hp` はこの意味で turn-invariant で、block は実際に防いだ分だけ
HP と 1:1 で評価されます（余剰 block と非攻撃 intent に対する block は 0 点。block は
ターンをまたいで持ち越されないため）。詳細は `decision/value.py` の docstring を参照。

## フォールバックと例外

次のケースでは Beam Search の結果に action が無いため
`HeuristicCombatSelector` へフォールバックします。

- root が現在の semantic scope 外
- `emulate_actions` batch が RL に reject され、かつ client session が引き続き fresh な場合
- 制限時間内に resolved actionable candidate を得られなかった
- continuation step limit までに macro-action が解決しなかった

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

time budget で continuation 解決途中に終了した場合、未解決 continuation node は親の
stale Value を継承していても actionable candidate には昇格しません。stable sibling があれば
その sibling だけが候補に残ります。

## Branch cleanup

探索で作った Branch は `finally` で cleanup します。

1. `cancel_branches`
2. client が引き続き fresh request を送れる状態なら `release_branches`

既知の recoverable な reject、または completion が確実に uncertain ではない transport failure は
best-effort cleanup としてログに残して続行します。一方、protocol error、faulted operation、
completion-uncertain transport error、予期しない実装例外は、正常な探索結果を返すために
握りつぶしません。

`pending_retry` または `session_invalid` が立った場合は、新しい cleanup request を送ると
session sequencing を壊すため中止します。

## 現在の既定実装について

`PriorHeuristicPolicy` と `HeuristicValueFunction` は学習済みモデルではなく、
モデル接続前でも pipeline を end-to-end で動かすための簡易実装です。
本番の意思決定品質を担保するものではありません。
