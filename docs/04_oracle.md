# Oracle collection

## 0. 文章の目的

この文書は Budgeted Oracle、Oracle JSONL、teacher/search-budget provenance、scenario harvesting を説明する。Value と learned `action_score` の consumer は [02_decision_core.md](02_decision_core.md)、stable-pruner target は [03_stable_pruner.md](03_stable_pruner.md) を参照する。

## 1. 概要

Oracle collection は runtime policy が訪れた decision state で、より広い/深い teacher search を実行して training target を記録する。`BudgetedOracleCollector` は root-action target と stable-pruning target を作り、`RootValueLoggingOracleCollector` は root action 直後の raw DTO と deeper target を対応付ける。

Oracle target は teacher class だけでなく search budget に依存するため、`oracle_teacher_provenance.py` は provenance と target-generation config をまとめて fingerprint し、意図しない混在を既定で拒否する。

`scenario_harvest.py` は Whole Run log から Combat 開始 scenario を作る。scenario JSON は wire payload のまま保ち、source-run provenance、GOD mode、completion、promotion eligibility、train/val/test split は `harvest_manifest.json` に分離する。

## 2. Architecture

`OracleCollectionConfig` は teacher search budget と target beam width を分ける。root は既定で `exhaustive_root_actions=True` なので available action 全件を評価する。learned `action_score` loader も既定でこの contract を要求する。

| target | 意味 |
|---|---|
| `RootActionOracleTarget` | root action ごとの RNG outcome を集約した estimated Q |
| `StableNodeOracleTarget` | descendant-derived `target_node_score` |
| `terminal` | exact terminal utility |
| `value_bootstrap` | fresh leaf の Value bootstrap |
| `no_target` | label にできない censored node |

current constants は `ORACLE_RECORD_SCHEMA_VERSION = 6`、`ORACLE_EPISODE_RESULT_SCHEMA_VERSION = 2`、`ORACLE_VALUE_MASK_VERSION = "1.2"`。Value/action-score consumer は wire schema、mask version、public `dto_version` の exact generation を検証する。

`TEACHER_PROVENANCE_SUMMARY_SCHEMA_VERSION = 2` は provenance と `oracle_targets.metadata` の `oracle_beam_width`、`target_beam_width`、`top_k_actions`、`max_depth`、`max_continuation_steps`、`time_budget_ms`、`exhaustive_root_actions`、`pruner_name`、`pruner_version`、`rng_sampling` を fingerprint する。`pruner_name`、`pruner_version`、`rng_sampling` は両 payload 間の一致も検証する。

scenario harvest は `HARVEST_MANIFEST_SCHEMA_VERSION = 1`。split は `source_run_sha256` から決めるため、同一 Whole Run の sibling combat は同じ split に入る。run completion は JSONL の最後の non-empty record が `self_play_run_result` かで判定する。途中に run-result record があっても、その後に record が続けば `source_completed=False` である。

`god_mode=True` は `god_mode_coverage` / `coverage_pretraining` で `promotion_eligible=False`。`god_mode=False` は `normal_policy` だが、`promotion_eligible=True` になるのは `source_completed=True` の場合だけである。GOD mode が不明なら `unknown` かつ promotion 対象外とする。

## 3. API

```python
@dataclass(frozen=True)
class OracleCollectionConfig:
    beam_config: BeamSearchConfig = field(
        default_factory=lambda: BeamSearchConfig(
            beam_width=32,
            top_k_actions=8,
            max_depth=4,
        )
    )
    target_beam_width: int = 8
    exhaustive_root_actions: bool = True
    rng_sampling: str = "independent"

class BudgetedOracleCollector:
    async def collect(self, instance_id: str, root_decision: Mapping[str, Any], *, timeout_s: float) -> OracleCollectionResult
```

```python
inspect_oracle_teacher_provenance(paths, *, allow_mixed_teachers=False) -> OracleTeacherProvenanceSummary
require_matching_teacher_provenance(
    artifact_summary,
    evaluation_summary,
    *,
    allow_teacher_mismatch=False,
) -> bool
```

```python
dto_to_scenario_spec(dto, *, seed: int) -> JsonObject | None
is_completed_run_log(path: Path) -> bool
harvest_scenario_records_from_jsonl(
    path: Path,
    *,
    exclude_final_combat: bool,
    rng: random.Random | None = None,
) -> list[JsonObject]
source_run_split(
    source_run_sha256: str,
    *,
    split_seed: int = 0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> str
```

## 4. 使用例

```bash
python -m sts2_training.runner.oracle_collection \
  --scenario data/scenarios/slime.json \
  --output data/combat_oracle/slime-001.jsonl \
  --oracle-beam-width 32 \
  --oracle-top-k 8 \
  --oracle-depth 4 \
  --target-beam-width 8
```

```bash
python -m sts2_training.runner.scenario_harvest \
  --input-dir data/runs \
  --output-dir data/scenarios \
  --split-seed 0 \
  --val-fraction 0.1 \
  --test-fraction 0.1
```

## 5. 補足説明

Oracle は ground truth ではなく teacher policy/value と target-generation config に依存する estimate である。learned `action_score` は `oracle_targets.root_actions[].estimated_q` を同一 decision 内 ranking の teacher として使い、runtime score を calibrated Q と解釈しない。

harvest の `promotion_eligible=False` は scenario を全 training から排除する意味ではなく、normal-policy の promotion/evaluation gate に混ぜないための sidecar metadata である。Value/action-score の label/filter contract は [02_decision_core.md](02_decision_core.md)、runner CLI は [07_runner_cli.md](07_runner_cli.md) を参照する。
