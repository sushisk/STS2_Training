# runner CLI

## 0. 文章の目的

この文書は `src/sts2_training/runner/` の実行入口を説明する。Oracle collection / scenario harvesting は [04_oracle.md](04_oracle.md) で扱い、ここでは self-play、episode runner、stable pruner 学習・RL・A/B、floor reach evaluation、start entrypoints、共通 CLI scaffold を対象にする。

## 1. 概要

`runner/` は API client、decision engine、scenario config、log writer をつなぐ実行層である。通常の run は `EpisodeRunner` が「get decision -> engine/selector で action 選択 -> commit action」を繰り返し、terminal または limit で `EpisodeResult` を返す。

CLI は大きく、from-scratch Whole Run を始める `start_new_run.py`、scenario から Combat を始める `start_combat_from_state.py`、複数 run を回す `self_play.py`、stable pruner の学習/評価系、到達 floor 評価に分かれる。`floor_reach_eval.py` は独立 seed の Whole Run ごとに最深 `totalFloor` を記録して集計する。`_cli.py` は host/port/log/search-mode などの共通引数と logging/result print を提供する。

## 2. Architecture

| ファイル | 役割 |
|---|---|
| `episode.py` | `build_engine()`、`EpisodeRunner`、`start_and_run()`、limit/timeout validation |
| `start_new_run.py` | `NewRunConfig` に相当する instance config で Whole Run を開始 |
| `start_combat_from_state.py` | `CombatScenario` JSON から Combat instance を開始 |
| `self_play.py` | seed batch を回し、run result と selection log を保存 |
| `stable_pruner_learn.py` | Oracle/RL logs を発見・正規化し、学習/eval tool を呼ぶ one-line entrypoint |
| `stable_pruner_rl.py` | fixed-seed paired A/B harness を使って on-policy RL trajectory を収集 |
| `stable_pruner_ab.py` | baseline pruner と learned pruner を同一 scenario/seed で比較 |
| `stable_pruner_ab_stats.py` | paired outcome の sign-test summary |
| `stable_pruner_ab_suite.py` | manifest による multi-scenario A/B |
| `floor_reach_eval.py` | floor 到達状況を tracking する evaluation |
| `rl_server_pool.py` | 評価中だけ paired `STS2_RL` TCP server を起動・停止する context manager |
| `_cli.py` | 共通 argparse / logging / result formatting |

`stable_pruner_learn.py` は `.jsonl`、`.json`、`.log`、`.txt` と directory/glob を受け、current Oracle/RL record を staged JSONL に抽出する。`--learn supervised|rl|auto` と `--start fresh|resume|auto`、`--data-mode auto-split|train|validate` を別々に解決する。

`add_common_arguments()` は `--host` の既定を `127.0.0.1`、`--port` を `8765`、`--connect-timeout` を `5.0`、`--decision-timeout` を `30.0`、`--max-decisions` と `--search-mode` を `None` にする。`--search-mode` を指定しない場合、共通 CLI 上では Beam Search は無効である。

## 3. API

```python
def build_engine(client, *, engine: CombatDecisionEngine | None = None, search_mode: str | BeamSearchConfig | None = None, beam_max_depth: int | None = None) -> CombatDecisionEngine

@dataclass(frozen=True)
class EpisodeResult:
    instance_id: str
    decisions_made: int
    final_dto: dict[str, Any]
    elapsed_s: float

class EpisodeRunner:
    async def run(self, instance_id: str, *, decision_timeout_s: float, max_decisions: int | None = None, close_timeout_s: float = 10.0) -> EpisodeResult

async def start_and_run(client, instance_config, *, start_timeout_s: float = 30.0, decision_timeout_s: float = 30.0, max_decisions: int | None = None, engine=None, search_mode=None, beam_max_depth=None) -> EpisodeResult
```

```python
async def start_new_run(client, *, character_id: str, ascension: int = 0, seed: int | None = None, ...) -> EpisodeResult
async def start_combat_from_state(client, scenario: CombatScenario, *, ...) -> EpisodeResult
async def run_self_play_batch(...) -> list[SelfPlayRunResult]
async def run_floor_reach_eval(...) -> list[FloorReachResult]
```

```python
@dataclass(frozen=True)
class FloorReachResult:
    run_id: str
    seed: int
    max_total_floor: int
    act_index_at_max: int | None
    decisions_made: int
    decision_source_counts: dict[str, int]
    outcome: str | None
    error: str | None
    elapsed_s: float

async def run_floor_reach_eval(
    *,
    character_id: str,
    num_runs: int,
    concurrency: int | None = None,
    ascension: int = 0,
    use_beam: bool = True,
    connection_factory: Callable[[], Any] | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    connect_timeout_s: float = 5.0,
    decision_timeout_s: float = 90.0,
    max_decisions: int | None = 600,
    search_mode: str | BeamSearchConfig | None = None,
    beam_max_depth: int | None = None,
    policy: PolicyModel | None = None,
    value_fn: ValueModel | None = None,
    stable_pruner: StableFrontierPruner | None = None,
    detailed_log_dir: Path | None = None,
    eval_epsilon: float = 0.0,
    ports: Sequence[int] | None = None,
) -> list[FloorReachResult]

summarize_floor_reach(results: list[FloorReachResult]) -> dict[str, Any]
```

```python
@dataclass(frozen=True)
class RlServer:
    port: int
    pid: int
    log_path: Path

def resolve_rl_root(root: str | os.PathLike[str] | None = None) -> Path
def free_ports(count: int, *, host: str = "127.0.0.1") -> list[int]

class RlServerPool:
    def __init__(
        self,
        *,
        ports: Sequence[int],
        root: str | os.PathLike[str] | None = None,
        host: str = "127.0.0.1",
        log_dir: str | os.PathLike[str] | None = None,
        startup_timeout_s: float = 180.0,
        max_message_bytes: int | None = None,
    ) -> None
```

```python
discover_log_files(inputs) -> tuple[Path, ...]
prepare_logs(inputs, *, staging_root: Path) -> PreparedLogs
resolve_learning_plan(...) -> LearningPlan
run_learning(args: argparse.Namespace) -> LearningRunSummary
```

## 4. 使用例

Whole Run:

```bash
python -m sts2_training.runner.start_new_run \
  --host 127.0.0.1 \
  --port 8765 \
  --character-id IRONCLAD \
  --seed 123 \
  --search-mode standard \
  --max-decisions 200
```

Combat scenario:

```bash
python -m sts2_training.runner.start_combat_from_state \
  --scenario data/scenarios/slime.json \
  --search-mode standard \
  --max-decisions 20
```

stable pruner 学習:

```bash
stable-pruner-learn data/combat_oracle \
  --learn supervised \
  --start fresh \
  --data-mode auto-split \
  --output tools/output/stable_pruner_weights.json
```

A/B:

```bash
python -m sts2_training.runner.stable_pruner_ab \
  --scenario data/scenarios/slime.json \
  --weights tools/output/stable_pruner_weights.json \
  --seeds 101,102,103 \
  --output tools/output/stable_pruner_ab.json
```

floor reach 評価（既に `127.0.0.1:8765` で RL server が起動している場合）:

```bash
python -m sts2_training.runner.floor_reach_eval \
  --character-id IRONCLAD \
  --num-runs 10 \
  --search-mode standard \
  --max-decisions 200 \
  --output data/evaluation/floor_reach.json
```

## 5. 補足説明

`run_floor_reach_eval()` は `ports` を渡すと、既定の `concurrency` をその port 数にする。1 台の RL server は request lock のため直列化されるので、実際に並列評価するには server ごとに別 port が必要である。`eval_epsilon` の既定は `0.0` で、fallback selector の探索を評価値へ混ぜない。`detailed_log_dir` を指定すると root board、action score、selection event、Beam score trace に加え、失敗 decision の `decision_failed` event も run ごとの JSONL に残る。

`RlServerPool` は `--rl-root` または `STS2_RL_ROOT` で paired checkout を解決し、context を抜けると起動した server とその worker process を停止する。Whole Run の RL 側 Combat 委譲、その評価 CLI の `--start-rl-servers` / `--turn-boundary-scoring`、ログの読み方は [whole_run_combat_delegation_20260822.md](whole_run_combat_delegation_20260822.md) を参照する。

`stable-pruner-learn` は内部で tool scripts を呼ぶため、学習依存を含めた install が必要である。runtime inference 自体は [03_stable_pruner.md](03_stable_pruner.md) の通り標準ライブラリで動く。Oracle collection CLI と scenario harvesting は [04_oracle.md](04_oracle.md)、live/replay でログを見る方法は [08_visualizer.md](08_visualizer.md) を参照する。
