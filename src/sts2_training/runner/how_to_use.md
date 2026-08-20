# runner の使い方

`sts2_training.runner` は、`sts2_training.decision.CombatDecisionEngine` の上に載る
**「instanceを起動して、終わるまで動かす」全体の管理役**です。2つの入り口（起動の仕方が
異なるだけ）と、それらが共有する1つの実行ループから構成されます。

```text
start_combat_from_state(CombatScenario)   完全な盤面を渡してCombatを開始
start_new_run(NewRunConfig)               通常のゲームスタート(deckなし、character+seedのみ)
        │
        ▼
   client.start_instance(instance_config)
        │
        ▼
   EpisodeRunner.run()   ← 2つの入り口が共有する唯一のループ
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
| `NewRunConfig` | 通常のゲームスタート | character_id のみ(ascension/seedはデフォルトあり) |

`CombatScenario`は**デフォルト値を持たない必須フィールド**を意図的に多く持っています。
「完全な盤面情報」を渡し忘れた場合、`RL`側に不完全なリクエストが届く前に、コンストラクタ
呼び出しの時点で`TypeError`になるようにするためです。`NewRunConfig`は盤面を渡さない
通常開始用なので、`ascension`と`seed`にデフォルトがあります。

Whole Runのsnapshot restoreは現時点ではサポートしていません。`STS2_RL`の
`WholeRunInstance`は`seed`/`character_id`/`ascension`からfresh runを開始するだけなので、
`AsyncTrainingApiClient.start_instance()`は`instance_type="whole_run"`の設定に
`snapshot_json`が含まれている場合、値が空や`None`でもwire送信前に拒否します。raw
`start_instance()`へ直接snapshotを渡してresumeしたつもりになる経路もfail closedです。

`CombatScenario`の出力はSTS2_RLの`build_scenario_from_spec()`
(`Common/schemas/combat_scenario_input_schema.json`)のwire shapeに合わせます。runner側では
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
  `EpisodeRunner.run`をまとめた本体。2つの`start_*`入り口はそれぞれの`instance_config`
  を組み立ててこれに委譲するだけです。

### `start_combat_from_state.py` / `start_new_run.py`

各モジュールは対応する`instance_config`の準備とCLIを持ちます。CLIの共通部分
(`--host`等の引数・結果のJSON出力)は`_cli.py`に集約されています。

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

2つの入り口はいずれも`search_mode`(プリセット名)と`beam_max_depth`(深さだけの上書き)
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

## 2つの入り口の使い分けまとめ

| 入り口 | いつ使うか |
|---|---|
| `start_combat_from_state` | 特定の戦闘盤面からの評価・デバッグ・特定シナリオでの学習 |
| `start_new_run` | 通常のフルラン(データ収集の基本形) |

## 評価の並列化(`--start-rl-servers` / `--ports`)

RL サーバ 1 台はクライアント側から並列化できない。`API/tcp_server.py` は
`_handler_lock` という **1 個の `asyncio.Lock` で全接続の全リクエストを直列化**しており、
その背後で Emulator は spawn された pythonnet/CLR プロセス 1 個の中で動く
(`API/api_runtime.py`)。したがって 1 ポートに対して `--concurrency` を上げても
キューが伸びるだけで速くならない。

CPU を使い切るには**サーバをポートの数だけ立てて、worker を 1 台ずつに固定する**。

いちばん簡単なのは `evaluate_whole_run.py` に起動と停止を任せる方法。ポートは自動で選ばれ、
終了時(Ctrl-C や例外を含む)にプロセスツリーごと停止する。

```
python tools/evaluate_whole_run.py --character-id IRONCLAD --num-runs 20 \
    --start-rl-servers 4 --rl-root C:\STS2_RL ...
```

checkout の場所は `--rl-root` か環境変数 `STS2_RL_ROOT` で渡す。環境変数を使う場合、
PowerShell は `$env:STS2_RL_ROOT = "C:\STS2_RL"`、cmd.exe は `set STS2_RL_ROOT=C:\STS2_RL`、
POSIX shell は `export STS2_RL_ROOT=/path/to/STS2_RL`。
PowerShell の `set` は `Set-Variable` のエイリアスで環境変数にはならないので注意。

サーバを自分で管理する場合は `--ports` を使う。この場合は開始前に全ポートへ疎通確認を行い、
listen していないポートを名指しして即座に失敗する(サーバが 1 台欠けていると、その worker の
担当分が丸ごと接続エラーになり、残った run だけで平均が出てしまうため)。

```
# サーバ側: 4 プロセス
python -m API.tcp_server --port 8765
python -m API.tcp_server --port 8766
python -m API.tcp_server --port 8767
python -m API.tcp_server --port 8768

# Training 側: worker がポートに 1 対 1 で張り付く(--concurrency は ports 数が既定)
python tools/evaluate_whole_run.py --character-id IRONCLAD --num-runs 20 \
    --ports 8765,8766,8767,8768 ...
```

`--concurrency` をポート数より大きくすると、複数 worker が 1 台を共有して
そのサーバのロックで直列化される。その場合は WARNING を出したうえで実行は継続する。

なお `emulate_actions` のバッチサイズは `beam_width × top_k_actions` が上限であり
(`max_batch_size=64` には届かない)、1 リクエストあたりの仕事量は既に飽和している。
1 戦あたりのリクエスト数は実測 322〜662 回、1 リクエスト 0.2〜0.6 秒。
短縮できるのは**戦を跨いだ並列化**だけである。
