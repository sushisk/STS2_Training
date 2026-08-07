# CombatDecisionEngine の使い方

`sts2_training.decision` は、`AsyncTrainingApiClient`(DTO v0.7)の上に載る
**Policy decision + Beam Search + Value function** の判断ロジック基盤です。
`selection.HeuristicCombatSelector`(1手だけ見てランダムに選ぶ)を置き換えるのではなく、
その上位に位置し、失敗時のフォールバック先として内部で再利用します。

## 全体の流れ

```text
get_decision(root)
    -> PolicyModel.propose_batch()     top_k候補を提案(盤面ごとに一括)
    -> emulate_actions([...])          1 Beam Level = 1 API request
    -> ValueModel.evaluate_batch()     結果盤面を一括評価
    -> (pruneして次のdepthへ、または終了)
    -> 最良のroot直下actionをcommit_action
```

`docs/STS2_next_implementation_plan.md` の設計方針通り、**1つのbeam深さ = 1回の
`emulate_actions` request** です。あるdepthで生き残っている全beam nodeの
policy候補をまとめて1つのbatchにして送るため、想定応答時間
(`beam search ~= 5ms + 1ms/decision`、盤面をまとめて流した分だけ償却される)
が成立します。

## 最小構成の使い方

```python
import asyncio

from sts2_training.api import AsyncTrainingApiClient, TcpConnection
from sts2_training.decision import CombatDecisionEngine


async def main() -> None:
    connection = TcpConnection(host="127.0.0.1", port=8765)
    async with AsyncTrainingApiClient(connection) as client:
        instance_id = await client.start_instance(
            {"instance_type": "combat"}, timeout_s=30.0
        )

        # instance / clientの寿命に対して1個だけ construct する。
        # BeamSearchEngineがbranch_id/rng_idのアロケータを保持しており、
        # instance生存期間中ずっとユニークである必要があるため。
        engine = CombatDecisionEngine(client)

        while True:
            response = await engine.decide_and_commit(instance_id, timeout_s=30.0)
            dto = response["masked_emulator_dto"]
            if not dto.get("legal_actions"):
                break


asyncio.run(main())
```

`decide_and_commit()` は内部で `get_decision -> beam search (or fallback) ->
commit_action` を行います。個別に制御したい場合は `decide()` を呼び、
`DecisionOutcome.chosen_action_id` を見てから自分で `commit_action` を呼んでください。

## Beam Search が働く条件

Beam Search(`emulate_actions` によるBranch分岐)は、**通常の戦闘Decision**
(`legal_actions` の `action_type` が `card`/`potion`/`system` のみ)でしか行いません。
`choice_card` や Map選択などRNG Hypothesis分岐が未対応のBoundaryでは、
`CombatDecisionEngine` は自動的に `HeuristicCombatSelector` にフォールバックします
(RL側 `instance_whole_run.py` の `fault_kind="rng_hypothesis_unsupported_at_boundary"`
を参照。Beam Search側からは絶対に踏まない設計です)。

`legal_actions` が1件だけの場合(強制手)は、Beam SearchもFallbackも呼ばず
そのまま採用します(`DecisionOutcome.source == "forced_single_action"`)。

## PolicyModel / ValueModel を差し替える

どちらも抽象基底クラスです。今のデフォルト実装(`PriorHeuristicPolicy` /
`HeuristicValueFunction`)は学習済みモデルが無くても動く簡易版で、
本格的な学習済みモデルに差し替える前提の土台です。

```python
from sts2_training.decision import ActionCandidate, PolicyModel, ValueModel


class MyPolicy(PolicyModel):
    def propose(self, legal_actions, masked_emulator_dto, *, top_k):
        ...  # 単一盤面用(デフォルトのpropose_batchはこれをループするだけ)

    def propose_batch(self, requests, *, top_k):
        ...  # 本命はこちら。beam1階層分の全盤面をまとめて1回で推論する。


class MyValue(ValueModel):
    def evaluate(self, masked_emulator_dto):
        ...

    def evaluate_batch(self, dtos):
        ...  # 本命はこちら。


engine = CombatDecisionEngine(client, policy=MyPolicy(), value_fn=MyValue())
```

`propose`/`evaluate` は単発呼び出し用の最小実装(デフォルトの `*_batch` は
これをループするだけ)です。実際の学習済みモデルは **`propose_batch` /
`evaluate_batch` を直接オーバーライドしてバッチ推論**してください。それが
`policy decision ~5ms`、`value function ~1ms` という想定応答時間の前提です。

## BeamSearchConfig の主なパラメータ

| パラメータ | 意味 |
|---|---|
| `beam_width` | 各depthで生き残らせるnode数 |
| `top_k_actions` | 1nodeあたりpolicyに提案させる候補数 |
| `max_depth` | 最大探索深さ(先読み手数) |
| `simulation_options` | `emulate_actions` に渡す停止条件(既定 `{"stop_condition": "next_decision"}`) |
| `time_budget_ms` | search呼び出し全体の時間予算(`timeout_s` と併用、短い方が優先) |
| `max_batch_size` | 1回の`emulate_actions` requestに含める最大item数(既定64、RLの`BranchManager.max_branches`標準値に合わせている)。1 depthの候補数がこれを超える場合は同じdepthで複数requestに分割して送る |
| `beam_searchable_action_types` | Beam Searchを許可する `action_type` の集合 |

## 後始末(Branch cleanup)

`BeamSearchEngine.search()` は探索中に作成した非rootBranchを、成功・失敗・
タイムアウトいずれの場合も `finally` で `cancel_branches` + `release_branches`
してから戻ります(`BeamSearchConfig.release_branches_on_finish=False` で無効化可能)。
クライアントに `pending_retry` が残っている場合(通信が不確定なまま終わった場合)は
後始末をスキップします。呼び出し側の通常のretry手順(`client.pending_retry` を見て
`retry_request()` する)が引き続き必要です。Beam Search自体はこの手順を代替しません。

## 既知の制約・今後の拡張ポイント

- `rng_id` はBeam Searchのroot直下の各候補actionごとに新規発行し、
  それ以深は契約通り親Branchと同じ`rng_id`を継承します
  (`LookaheadSearcher`のようなroot action毎の複数Hypothesis平均化は未実装)。
- 現在の`PriorHeuristicPolicy`/`HeuristicValueFunction`は学習済みモデルではなく、
  `selection.HeuristicCombatSelector`と同じ「初期段階の簡易ロジック」です。
- `emulate_actions`はRL側`BranchManager.poll()`が同期的に全Branchを解決してから
  応答するため(v0.7契約)、`branch_results`は`completed`/`partial`/`faulted`のみで
  `queued`/`running`は返らず、ポーリングは不要です。1 depthの候補数が
  `max_batch_size`を超える場合のみ、同じdepthを複数requestへ分割して送ります。
  分割中の1 requestが拒否された場合、それより前に成功したrequestの結果は保持したまま
  それ以上の分割送信・深掘りを打ち切ります(`BeamSearchResult.reason`に
  `"emulate_actions_rejected:..."`が入ります)。
