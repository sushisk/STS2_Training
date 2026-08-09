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
   terminal/run_terminal の明示的な終端に到達したら close_instance して EpisodeResult を返す
```

`instance_config`の組み立て方が違うだけで、起動後にinstanceを終端まで動かす処理は
`instance_type`によらず全く同じです。`CombatDecisionEngine`自体が既にこれを吸収して
いて(`card`/`potion`/`system`以外はheuristic fallbackに自動的に委譲し、fallback側は
`action_type`文字列で汎用的に分類するため、Whole Runの`map_select`/`event_choice`/
`shop_choice`/`rest_choice`/`reward_select`境界でも問題なく動きます)、`runner`パッケージ
はこの上に新しい判断ロジックを一切追加していません。ループとライフサイクル管理のみです。

## モジュール構成

### `scenario.py` — `instance_config`の組み立て

| クラス | 用途 | 必須フィールド |
|---|---|---|
| `CombatScenario` | 特定盤面からCombat開始 | character_id, player_hp, player_max_hp, hand, draw_pile, discard_pile, enemies |
| `RunSnapshot` | 特定盤面からrun再開 | character_id, ascension, seed, snapshot_json |
| `NewRunConfig` | 通常のゲームスタート | character_id のみ(ascension/seedはデフォルトあり) |

`CombatScenario`/`RunSnapshot`は**デフォルト値を持たない必須フィールド**を意図的に
多く持っています。「完全な盤面情報」を渡し忘れた場合、`RL`側に不完全なリクエストが
届く前に、コンストラクタ呼び出しの時点で`TypeError`になるようにするためです。
`CombatScenario`の出力はSTS2_RLの`build_scenario_from_spec()`(
`Common/schemas/combat_scenario_input_schema.json`)のwire shapeに合わせます。runner側では
`potions=["FIRE_POTION", None, "BLOCK_POTION"]`のようなbelt順の短縮形を
`[{"slot": 0, "potion_id": ...}, {"slot": 2, "potion_id": ...}]`へ、
`player_powers={"STRENGTH": 2}`や`EnemyScenario(..., powers={...})`を
`[{"power_id": "STRENGTH", "amount": 2}]`へ変換します。すでにRL shapeのmapping/listを
持っている場合はそのexact formも渡せます。モデル化していないフィールド
(`pending_choice`、カード個別のアップグレード情報等)は`extra`引数でそのままマージできます。

### `episode.py` — 共有の実行ループ

- `EpisodeRunner.run(instance_id, decision_timeout_s=...)`: Combatの`terminal: true`または
  Whole Runの`run_terminal: true`が明示されるまで`decide_and_commit()`を繰り返し、成功・
  失敗・タイムアウトいずれの場合も`finally`で`close_instance`をベストエフォートで呼びます
  (`client`に`pending_retry`/`session_invalid`が残っている場合はスキップ)。非終端なのに
  selectableな`legal_actions`が空なら成功扱いにはせず`NoAvailableActionError`でfail closedします。
  戻り値`EpisodeResult`の`final_dto`が最後に観測した`masked_emulator_dto`です - 勝敗は
  `final_dto["outcome"]`を直接読んでください。暴走防止用に`max_decisions`を指定でき、
  超過すると`EpisodeLimitExceeded`が送出されます(それでも`close_instance`は呼ばれます)。
- `build_engine(client, *, engine=None, search_mode=None, beam_max_depth=None)`:
  `search_mode`/`beam_max_depth`から`CombatDecisionEngine`を組み立てる小さなヘルパー。
- `start_and_run(client, instance_config, ...)`: `build_engine` + `start_instance` +
  `EpisodeRunner.run`をまとめた本体。3つの`start_*`入り口はそれぞれの`instance_config`
  を組み立ててこれに委譲するだけです。

### `start_combat_from_state.py` / `start_run_from_state.py` / `start_new_run.py`

各モジュールは対応する`instance_config`の準備とCLIを持ちます。現在未対応の
`start_run_from_state`だけは、誤って新規runを開始しないため`start_instance`へ委譲せず
明示的に失敗します。CLIの共通部分(`--host`等の引数・結果のJSON出力)は`_cli.py`に
集約されています。

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

## 探索モードの選択(`search_mode` / `beam_max_depth`)

3つの入り口すべてが`search_mode`(プリセット名)と`beam_max_depth`(深さだけの上書き)
を受け取れます。内部で`decision.search_modes.resolve_search_mode`を呼び、
`CombatDecisionEngine`に渡す`BeamSearchConfig`を組み立てます(プリセット一覧は
`decision/how_to_use.md`参照)。

```python
result = await start_combat_from_state(
    client, scenario, decision_timeout_s=30.0,
    search_mode="deep",       # プリセットを選ぶ
    beam_max_depth=5,         # そのプリセットのmax_depthだけ上書き(省略可)
)
```

```sh
python -m sts2_training.runner.start_new_run \
    --host 127.0.0.1 --port 8765 --character-id IRONCLAD \
    --search-mode deep --beam-depth 5
```

**完全に手組みしたい場合**は`engine`引数(既に構築済みの`CombatDecisionEngine`)を
渡してください。`engine`と`search_mode`/`beam_max_depth`を同時に渡すとエラーに
なります(`episode.build_engine`が検証 - どちらが実際に効くのか曖昧なまま黙って
片方を無視することを避けるため)。

```python
from sts2_training.decision import CombatDecisionEngine, resolve_search_mode

my_engine = CombatDecisionEngine(client, beam_config=resolve_search_mode("wide", max_depth=6))
result = await start_combat_from_state(client, scenario, engine=my_engine)
```

## `start_run_from_state`の既知の制約(現状未対応)

STS2_RLの`API/instance_whole_run.py`の`WholeRunInstance`は、`instance_config`から
snapshotを読み取る仕組みをまだ持っていません(`WholeRunSession.load_state()`自体は
存在しますが配線されていません)。そのため`start_run_from_state()`は呼ぶと必ず
`RunSnapshotRestoreNotSupportedError`を送出します - 「本当は新規runなのに再開したと
誤解してしまう」事故を防ぐため、黙って新規runを開始する代わりに明示的に失敗させて
います。CLIもこのguardをローカルで発火させ、未対応の間はTCP接続を試みません。
STS2_RL側の対応が入ったら、このguardを外して`start_and_run`へ接続するのが残作業です。

## 3つの入り口の使い分けまとめ

| 入り口 | いつ使うか | 現状 |
|---|---|---|
| `start_combat_from_state` | 特定の戦闘盤面からの評価・デバッグ・特定シナリオでの学習 | 動作可能 |
| `start_run_from_state` | 特定のrun状態(周回中の特定地点)からの再開 | STS2_RL側の対応待ち |
| `start_new_run` | 通常のフルラン(データ収集の基本形) | 動作可能 |
