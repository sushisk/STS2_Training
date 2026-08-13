# runner CLI

## 0. 文章の目的

この文書は `src/sts2_training/runner/` の実行入口を説明する。Oracle collection / scenario harvesting は [04_oracle.md](04_oracle.md) で扱い、ここでは self-play、episode runner、stable pruner 学習・RL・A/B、floor reach evaluation、start entrypoints、共通 CLI scaffold を対象にする。

## 1. 概要

`runner/` は API client、decision engine、scenario config、log writer をつなぐ実行層である。通常の run は `EpisodeRunner` が「get decision -> engine/selector で action 選択 -> commit action」を繰り返し、terminal または limit で `EpisodeResult` を返す。

CLI は大きく、from-scratch Whole Run を始める `start_new_run.py`、scenario から Combat を始める `start_combat_from_state.py`、複数 run を回す `self_play.py`、stable pruner の学習/評価系、到達 floor 評価に分かれる。`_cli.py` は host/port/log/search-mode などの共通引数と logging/result print を提供する。

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
| `_cli.py` | 共通 argparse / logging / result formatting |

`stable_pruner_learn.py` は `.jsonl`、`.json`、`.log`、`.txt` と directory/glob を受け、current Oracle/RL record を staged JSONL に抽出する。`--learn supervised|rl|auto` と `--start fresh|resume|auto`、`--data-mode auto-split|train|validate` を別々に解決する。

## 3. API

```python
def build_engine(client, *, search_mode: str | None = None, beam_config: BeamSearchConfig | None = None, stable_pruner: StableFrontierPruner | None = None) -> CombatDecisionEngine

@dataclass(frozen=True)
class EpisodeResult:
    instance_id: str
    decisions: int
    terminal: bool
    final_response: dict[str, Any] | None

class EpisodeRunner:
    async def run(self) -> EpisodeResult

async def start_and_run(client, instance_config, *, engine=None, max_decisions=None, ...) -> EpisodeResult
```

```python
async def start_new_run(client, *, character_id: str, ascension: int = 0, seed: int | None = None, ...) -> EpisodeResult
async def start_combat_from_state(client, scenario: CombatScenario, *, ...) -> EpisodeResult
async def run_self_play_batch(...) -> list[SelfPlayRunResult]
async def run_floor_reach_eval(...) -> list[FloorReachResult]
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

## 5. 補足説明

`stable-pruner-learn` は内部で tool scripts を呼ぶため、学習依存を含めた install が必要である。runtime inference 自体は [03_stable_pruner.md](03_stable_pruner.md) の通り標準ライブラリで動く。Oracle collection CLI と scenario harvesting は [04_oracle.md](04_oracle.md)、live/replay でログを見る方法は [08_visualizer.md](08_visualizer.md) を参照する。
