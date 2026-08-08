# runner の使い方

`sts2_training.runner` は、`sts2_training.decision.CombatDecisionEngine` の上に載る
**「instanceを起動して、終わるまで動かす」全体の管理役**です。3つの入り口(起動の仕方が
異なるだけ)と、それらが共有する1つの実行ループから構成されます。

```text
start_combat_from_state(CombatScenario)   完全な盤面を渡してCombatを開始
start_run_from_state(RunSnapshot)         完全な盤面を渡してrunを再開(現状未対応、後述)
start_new_run(NewRunConfig)               通常のゲームスタート(deckなし、character+seedのみ)
        │
        ▼
   client.start_instance(instance_config)
        │
        ▼
   EpisodeRunner.run()   ← 3つの入り口が共有する唯一のループ
        │  decide() -> commit_action() を繰り返す
        │  (CombatDecisionEngineが内部でbeam search / heuristic fallbackを選択)
        ▼
   終端(legal_actionsが空)に到達したら close_instance して EpisodeResult を返す
```

`instance_config`の組み立て方が違うだけで、起動後にinstanceを終端まで動かす処理は
`instance_type`によらず全く同じです。`CombatDecisionEngine`自体が既にこれを吸収して
いて(`card`/`potion`/`system`以外はheuristic fallbackに自動的に委譲し、fallback側は
`action_type`文字列で汎用的に分類するため、Whole Runの`map_select`/`event_choice`/
`shop_choice`/`rest_choice`/`reward_select`境界でも問題なく動きます)、`runner`パッケージ
はこの上に新しい判断ロジックを一切追加していません。ループとライフサイクル管理のみです。

## 4つのモジュール

### `scenario.py` — `instance_config`の組み立て

| クラス | 用途 | 必須フィールド |
|---|---|---|
| `CombatScenario` | 特定盤面からCombat開始 | character_id, player_hp, player_max_hp, hand, draw_pile, discard_pile, enemies |
| `RunSnapshot` | 特定盤面からrun再開 | character_id, ascension, seed, snapshot_json |
| `NewRunConfig` | 通常のゲームスタート | character_id のみ(ascension/seedはデフォルトあり) |

`CombatScenario`/`RunSnapshot`は**デフォルト値を持たない必須フィールド**を意図的に
多く持っています。「完全な盤面情報」を渡し忘れた場合、`RL`側に不完全なリクエストが
届く前に、コンストラクタ呼び出しの時点で`TypeError`になるようにするためです。
`CombatScenario`のフィールドはSTS2_RLの`build_scenario_from_spec()`(
`Common/schemas/combat_scenario_input_schema.json`)に対応しており、モデル化していない
フィールド(pending_choice、カード個別のアップグレード情報等)は`extra`引数でそのまま
マージできます。

### `episode.py` — 共有の実行ループ(`EpisodeRunner`)

`run(instance_id, decision_timeout_s=...)`が本体です。`legal_actions`が空になる
(Combat勝敗確定、またはWhole Runの`run_terminal`)まで`decide()`→`commit_action()`を
繰り返し、成功・失敗・タイムアウトいずれの場合も`finally`で`close_instance`を
ベストエフォートで呼びます(`BeamSearchEngine._cleanup`と同じ設計思想 - `client`に
`pending_retry`/`session_invalid`が残っている場合はスキップ)。

戻り値`EpisodeResult`には`decisions_made`(実際にcommitした回数)、`final_dto`(最後に
観測した`masked_emulator_dto` - 勝敗は`final_dto["outcome"]`を直接読んでください、
STS2_RLの`agent/expose-terminal-outcome`以降はCombat/Whole Run双方で信頼できます)、
`decision_sources`(`beam_search`/`heuristic_fallback`/`forced_single_action`/`none`
それぞれが何回選ばれたかのカウント - beam searchがどれだけ機能しているかの簡易的な
健全性指標)が入っています。

暴走防止用に`max_decisions`を指定でき、超過すると`EpisodeLimitExceeded`が送出されます
(それでも`close_instance`は呼ばれます)。

### `start_combat_from_state.py` / `start_run_from_state.py` / `start_new_run.py`

各モジュールは「対応する`instance_config`を組み立てて`start_instance`し、
`EpisodeRunner`に渡す」という薄い管理役と、それを呼び出すCLIの両方を持っています。

```python
from sts2_training.api import AsyncTrainingApiClient, TcpConnection
from sts2_training.runner import CombatScenario, EnemyScenario, start_combat_from_state

scenario = CombatScenario(
    character_id="IRONCLAD",
    player_hp=50, player_max_hp=80,
    hand=["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"],
    draw_pile=[], discard_pile=[],
    enemies=[EnemyScenario(monster_id="CALCIFIED_CULTIST", hp=48)],
)

async with AsyncTrainingApiClient(TcpConnection(host="127.0.0.1", port=8765)) as client:
    result = await start_combat_from_state(client, scenario, decision_timeout_s=30.0)
    print(result.decisions_made, result.final_dto.get("outcome"))
```

CLIからも同じことができます(`--scenario`はJSONファイルパス、キーは`CombatScenario`の
フィールド名と一致させてください):

```sh
python -m sts2_training.runner.start_combat_from_state \
    --host 127.0.0.1 --port 8765 --scenario scenario.json
```

`start_new_run`は`seed`省略時にランダムなseedを自動生成します(呼ぶたびに違うrunに
なる、通常プレイと同じ挙動)。特定seedを固定したい場合のみ`--seed`を指定してください。

## `start_run_from_state`の既知の制約(現状未対応)

STS2_RLの`API/instance_whole_run.py`の`WholeRunInstance`は、`instance_config`から
snapshotを読み取る仕組みをまだ持っていません(`WholeRunSession.load_state()`自体は
存在しますが配線されていません)。そのため`start_run_from_state()`は呼ぶと必ず
`RunSnapshotRestoreNotSupportedError`を送出します - 「本当は新規runなのに再開したと
誤解してしまう」事故を防ぐため、黙って新規runを開始する代わりに明示的に失敗させて
います。設定の組み立て・CLIの配線自体は完成しているので、STS2_RL側の対応が入り次第、
このガードを外すだけで動くようになります。

## 3つの入り口の使い分けまとめ

| 入り口 | いつ使うか | 現状 |
|---|---|---|
| `start_combat_from_state` | 特定の戦闘盤面からの評価・デバッグ・特定シナリオでの学習 | 動作可能 |
| `start_run_from_state` | 特定のrun状態(周回中の特定地点)からの再開 | STS2_RL側の対応待ち |
| `start_new_run` | 通常のフルラン(データ収集の基本形) | 動作可能 |
