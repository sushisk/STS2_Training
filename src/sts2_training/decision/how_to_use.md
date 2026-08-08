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

## 4つのコンポーネントと責任範囲

このパッケージは4つのファイルに分かれており、それぞれの責任は明確に分離されています。
「提案する(Policy)」「評価する(Value)」「探索を組み立てる(BeamSearch)」
「優先順位を決める(Engine)」という機能軸で切られていると考えると見通しが良いです。

```text
CombatDecisionEngine ──uses──> BeamSearchEngine ──uses──> PolicyModel (propose_batch)
        │                              │            └──uses──> ValueModel (evaluate_batch)
        │                              └──uses──> client (get_decision/emulate_actions/cancel_branches/release_branches)
        └──fallback──> HeuristicCombatSelector (selection/heuristic_selector.py, 既存)
```

### `policy.py` — 「次に何を試すべきか」の提案責任

- **`PolicyModel`(抽象基底)**: 1つの盤面(`legal_actions` + `masked_emulator_dto`)に
  対して、有望そうな候補actionをbest-firstでtop_k個返すことだけが責任。勝率を
  判断する責任は一切持たない(それは`ValueModel`の仕事)。
  - `propose()`(抽象): 単一盤面用の最小契約。実装必須。
  - `propose_batch()`(具象): 複数盤面を1回で捌く本命インターフェース。既定実装は
    `propose()`をループするだけの素朴なものだが、**`BeamSearchEngine`は常に
    `propose_batch`しか呼ばない**。学習済みモデルを繋ぐ側は、これを直接
    オーバーライドしてバッチ推論に対応するのが本来の責任(~5ms/batchの
    latency予算はこの前提で設計されている)。
- **`ActionCandidate`(データクラス)**: `action_id`/`prior`/`action_type`の3つだけを
  持つPolicy出力の型。探索は`prior`ではなく`ValueModel`のスコアで選抜するため、
  `prior`はログ・デバッグ用の付随情報という位置づけ。
- **`PriorHeuristicPolicy`(既定実装)**: 学習済みモデルが無くても動かすための
  プレースホルダー。`HeuristicCombatSelector`と同じカテゴリ優先度
  (`card > choice_card > choice_confirm > choice_skip > その他`)で`legal_actions`を
  並べ替え、上位top_k件を返すだけで、強さ・勝率の予測は一切行わない。`rng`を渡すと
  カテゴリ内シャッフルが有効になり、探索の多様性を出せる。

### `value.py` — 「この盤面はどれだけ良いか」の評価責任

- **`ValueModel`(抽象基底)**: 1つの`masked_emulator_dto`をスカラー値に変換する
  ことだけが責任。actionを選ぶ責任は持たない(beam探索側がこの値でソート・剪定する)。
  - `evaluate()`(抽象) / `evaluate_batch()`(具象、既定は`evaluate`のループ)。
    `BeamSearchEngine`は常に`evaluate_batch`のみを呼ぶ。
- **`HeuristicValueFunction`(既定実装)**: 戦闘DTOから即席の特徴量を抜き出し、
  固定重みの線形和にするだけで、**デッキ構成やカード内容は一切見ない**
  (デッキ/カード評価とは完全に別レイヤー)。
  - `evaluate()`: まず勝敗が確定しているか(`_terminal_outcome`)を見て、
    確定していれば`±100,000`(`victory_bonus`/`defeat_penalty`)を即返す。
    **「勝敗が確定した状態は、他のどんな特徴量スコアよりも常に優先される」**
    という不変条件をここで保証している。非終端なら`_extract_features()`の加重和。
  - `_extract_features()`: HP比率・ブロック・敵HP比率・被弾予測ダメージ・
    生存敵数・バフデバフ合計、の6特徴量を計算するだけの純粋関数的責任。
    重み付けは`evaluate()`側の責任で、ここでは行わない。
  - `DEFAULT_WEIGHTS`はモジュールレベル定数として外出しされており、`weights`引数で
    上書き可能。「重みのチューニング」と「特徴量の定義」の責任を分けている。

### `beam_search.py` — 探索そのものの責任(このパッケージの核)

- **`BeamSearchConfig`**: 探索パラメータの束。妥当性検証込みで値を保持するだけ。
  `beam_searchable_action_types`(既定`{system, card, potion}`)がRL側のRNG
  Hypothesis対応範囲との整合性を握る唯一のフィールド。
- **`BeamNode`**: 探索木の1ノードのイミュータブルなスナップショット。`branch_id`/
  `parent_branch_id`/`rng_id`でRL側のBranch系譜を追跡し、`root_action_id`で
  「このノードはroot直下のどのactionから派生したか」を保持する——これが最終的に
  `decide()`が返す"root action"を特定する唯一の手がかり。
- **`BranchIdAllocator`**: instance生存期間中ずっとユニークな`branch_id`/`rng_id`を
  発行し続けることだけが責任(RL/Training契約上、branch_idはinstance内で生涯一意で
  cancel/release後も再利用不可)。**`BeamSearchEngine`インスタンスにつき1個だけ生成し、
  search()呼び出しをまたいで使い回す**という寿命管理が前提。
- **`BeamSearchEngine`**: `search()`は4段階のヘルパーをdepthごとに順に呼ぶ
  オーケストレーション責任のみを持ち、個々のロジックの詳細には立ち入らない。
  1. `_propose_frontier()`: 現beamの全ノード分の`legal_actions`をまとめて
     `PolicyModel.propose_batch`に渡し、各候補actionに新規`branch_id`/`rng_id`を
     発行して`emulate_actions`用アイテムを組み立てる。`rng_id`の継承ルール
     (root直下候補は新規発行、それより深いノードは親の`rng_id`を継承)もここに
     埋め込まれている。
  2. `_emulate_depth_batch()`: 1 depth分のアイテムを`max_batch_size`(既定64)ごとに
     分割し、複数回の`emulate_actions`として送る。チャンクが拒否された場合、その
     チャンクのbranch_idは「RL側に存在しないので後始末対象に含めない」判断も
     ここにあり、それより前に成功したチャンクの結果は保持したまま打ち切る
     (部分成功の扱い)。
  3. `_score_frontier()`: `branch_results`から解決済み(`completed`/`partial`)のものだけ
     抽出し、`ValueModel.evaluate_batch`で一括評価して新規`BeamNode`を構築する。
     「beamに残すか`finished`に確定させるか」の判定(終端状態/最終depth到達/
     `decision_point_id`が無い/`partial`かつ`expand_partial=False`)もここに集約。
  4. `_cleanup()`: 探索中に作った非rootBranchを`cancel_branches`+`release_branches`
     する後始末。`search()`本体の`try/finally`から**成功・失敗・タイムアウトいずれの
     経路でも必ず呼ばれる**。`client.pending_retry`/`session_invalid`が立っている
     場合はスキップする判断もここに閉じている。

### `engine.py` — 最上位のオーケストレーションと安全性の責任

- **`DecisionOutcome`**: 1回の意思決定の結果を表すイミュータブルな値オブジェクト。
  `source`フィールド(`"beam_search" | "heuristic_fallback" | "forced_single_action" |
  "none"`)が「どの経路で選ばれたか」を呼び出し側に必ず開示する責任を担う。
- **`CombatDecisionEngine`**: 「beam探索を試み、ダメならheuristicにフォールバックする、
  という意思決定の優先順位を管理すること」に責任が限定されており、探索アルゴリズムの
  中身にもHTTP通信の中身にも立ち入らない(それぞれ`BeamSearchEngine`と`client`に委譲)。
  - `__init__`: `policy`/`value_fn`が未指定ならデフォルト(`PriorHeuristicPolicy`/
    `HeuristicValueFunction`)を使う。**clientごとに1個だけ構築し使い回す**という
    寿命管理が前提(`BeamSearchEngine`が`BranchIdAllocator`を持つため)。
  - `decide()`: `get_decision`で盤面取得 →`legal_actions`が0件なら`"none"`即返し
    → 1件だけなら探索もfallbackも呼ばず`"forced_single_action"`即採用(無駄なAPI
    呼び出しを避ける最適化)→ それ以外は`_try_beam_search()`を試し、成功すれば採用、
    失敗すれば`HeuristicCombatSelector`にフォールバック。
  - `_try_beam_search()`: 例外処理方針の核心。`TransportError`(セッション状態が
    壊れた可能性がある通信エラー)は絶対に握りつぶさず再送出する(呼び出し元=
    clientを所有する側しか`retry_request()`等の復旧判断ができないため)。それ以外の
    例外(`PolicyModel`/`ValueModel`のバグ含む)はログを出して`None`を返し、意思決定
    ループ自体は止めない。
  - `decide_and_commit()`: `decide()`の結果を`commit_action`まで一気に進める薄い
    ラッパー。`chosen_action_id`が`None`なら`NoAvailableActionError`を送出する
    責任のみを追加。

補足: `beam_searchable_action_types`のチェックは`CombatDecisionEngine.decide()`と
`BeamSearchEngine.search()`の両方に存在する。これは安全性のための二重チェックでは
なく、`decide()`側の早期リターンは「対象外と分かっている呼び出しで無駄な往復を
避ける」ための最適化であり、実際の安全性は`search()`側の再チェックが担保している。

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
