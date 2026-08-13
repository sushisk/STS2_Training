# Oracle collection

## 0. 文章の目的

この文書は Budgeted Oracle と、その JSONL 出力、root-action value sample、teacher/search-budget provenance、scenario runner/harvester を説明する。対象ファイルは `decision/oracle_search.py`、`oracle_log.py`、`oracle_value_logging.py`、`oracle_teacher_provenance.py`、`runner/oracle_collection.py`、`runner/scenario.py`、`runner/scenario_harvest.py` である。Value 学習側の consumer は [02_decision_core.md](02_decision_core.md) を参照する。

## 1. 概要

Oracle collection は、runtime policy が訪れた decision state ごとに、より広い/深い Beam Search teacher を走らせる data collection path である。runtime に commit する action は通常の `CombatDecisionEngine` が選び、Oracle はその直前に training target を記録する。

`BudgetedOracleCollector` は root action target と stable-pruning target を作る。`RootValueLoggingOracleCollector` はさらに root action 直後の raw DTO と deeper target を対応付けた `RootActionValueSample` を保存する。JSONL writer は `combat_oracle_decision` と episode result record を append-only で書く。

Oracle target は teacher class だけでなく search budget にも依存する。そのため `oracle_teacher_provenance.py` は teacher provenance と target-generation config を合わせて fingerprint し、意図しない teacher/budget 混在を既定で拒否する。

## 2. Architecture

`OracleCollectionConfig` は teacher search budget と student/runtime target budget を分ける。root actions は既定で `exhaustive_root_actions=True` により available action 全件を teacher policy に要求して評価する。`exhaustive_root_actions=False` にすると root も通常の policy/top-k 制限に従うため、未提案の legal action は評価対象にならない。

target semantics は次の通りである。

| target | 意味 |
|---|---|
| `RootActionOracleTarget` | root action ごとに RNG hypothesis outcome を集約した estimated Q |
| `StableNodeOracleTarget` | stable-prune trace node に対する descendant-derived `target_node_score` |
| `terminal` | `ValueModel.exact_terminal_utility()` が明示的に exact value を返した terminal |
| `value_bootstrap` | fresh leaf を `ValueModel` で評価した bootstrap |
| `no_target` | Oracle が follow-up できず label にできない censored node |

`oracle_log.py` の current constants は `ORACLE_RECORD_SCHEMA_VERSION = 6`、`ORACLE_EPISODE_RESULT_SCHEMA_VERSION = 2`、`ORACLE_VALUE_MASK_VERSION = "1.2"`。`oracle_value_dto_contract()` は wire schema version、mask version、public `dto_version` を記録し、Value data consumer は 1 dataset 内で exact DTO generation が揃うことを検証する。

`oracle_teacher_provenance.py` の `TEACHER_PROVENANCE_SUMMARY_SCHEMA_VERSION = 2` は、record の `provenance` と `oracle_targets.metadata` の target-generation config を合わせて canonical JSON fingerprint にする。target-generation config には `oracle_beam_width`、`target_beam_width`、`top_k_actions`、`max_depth`、`max_continuation_steps`、`time_budget_ms`、`exhaustive_root_actions`、`pruner_name`、`pruner_version`、`rng_sampling` が含まれる。`pruner_name`、`pruner_version`、`rng_sampling` は provenance 側と metadata 側の一致も fail closed で確認する。

学習 artifact と held-out evaluation の teacher/search-budget set は `require_matching_teacher_provenance()` で照合できる。schema v1 の既存 provenance summary は provenance-only fingerprint として互換比較するが、新しい schema v2 artifact は target-generation config の差も mismatch として扱う。

`runner/scenario.py` は Combat start scenario の dataclass/JSON 変換を持ち、`runner/scenario_harvest.py` は completed run log から combat start DTO を scenario spec に変換する。

## 3. API

```python
@dataclass(frozen=True)
class OracleCollectionConfig:
    beam_config: BeamSearchConfig = field(default_factory=lambda: BeamSearchConfig(beam_width=32, top_k_actions=8, max_depth=4))
    target_beam_width: int = 8
    exhaustive_root_actions: bool = True
    rng_sampling: str = "independent"

class BudgetedOracleCollector:
    @classmethod
    def from_beam_engine(cls, engine: BeamSearchEngine, *, config: OracleCollectionConfig | None = None, stable_pruner: StableFrontierPruner | None = None) -> "BudgetedOracleCollector"
    async def collect(self, instance_id: str, root_decision: Mapping[str, Any], *, timeout_s: float) -> OracleCollectionResult
```

```python
oracle_collection_record(root_decision, result, *, instance_id, decision_index, runtime_transition, training_commit=None) -> dict[str, Any]
oracle_episode_result_record(*, instance_id, decisions_collected, final_dto, final_decision_metadata, completed, termination_reason, elapsed_s) -> dict[str, Any]

class OracleJsonlWriter:
    def write(self, root_decision, result, *, instance_id, decision_index, runtime_transition, training_commit=None) -> dict[str, Any]
    def write_episode_result(self, *, instance_id, decisions_collected, final_dto, final_decision_metadata, completed, termination_reason, elapsed_s) -> dict[str, Any]
```

```python
inspect_oracle_teacher_provenance(
    paths,
    *,
    allow_mixed_teachers=False,
) -> OracleTeacherProvenanceSummary
require_matching_teacher_provenance(
    artifact_summary,
    evaluation_summary,
    *,
    allow_teacher_mismatch=False,
) -> bool
```

## 4. 使用例

Oracle collection CLI:

```bash
python -m sts2_training.runner.oracle_collection \
  --scenario data/scenarios/slime.json \
  --output data/combat_oracle/slime-001.jsonl \
  --search-mode standard \
  --oracle-beam-width 32 \
  --oracle-top-k 8 \
  --oracle-depth 4 \
  --target-beam-width 8
```

scenario harvesting:

```bash
python -m sts2_training.runner.scenario_harvest \
  --input-dir data/runs \
  --output-dir data/scenarios
```

## 5. 補足説明

Oracle は ground truth ではなく、特定 teacher policy/value と target-generation config に依存した configuration-dependent estimate である。異なる search budget を同じ teacher label set とみなすと supervised objective 自体が変わるため、schema v2 provenance summary はその差を fingerprint に含める。

現在のデータ処理では `ORACLE_RECORD_SCHEMA_VERSION = 6` を前提にする。stable pruner training との接続は [03_stable_pruner.md](03_stable_pruner.md)、Value/trajectory consumer は [02_decision_core.md](02_decision_core.md)、CLI 全体は [07_runner_cli.md](07_runner_cli.md) を参照する。
