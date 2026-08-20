# decision core

## 0. 文章の目的

この文書は `decision/` の中核である Beam Search、Policy、Value、CombatDecisionEngine、Combat DTO 正規化、event branch search、candidate coverage に加え、learned Value、learned `action_score`、それらの training/data contract を説明する。stable frontier pruning の学習部分は [03_stable_pruner.md](03_stable_pruner.md)、Oracle target と collection は [04_oracle.md](04_oracle.md) に分ける。

## 1. 概要

decision core は、1 つの Combat decision DTO から action を選ぶ層である。`CombatDecisionEngine` はまず event choice の特殊処理を試し、Combat search 可能な状態なら `BeamSearchEngine` を実行し、actionable な search result がない場合は fallback selector に戻す。

`BeamSearchEngine` は `PolicyModel` が提案した候補 action を `AsyncTrainingApiClient.emulate_action(s)` で分岐実行し、`ValueModel` が stable/terminal state を評価し、beam width と depth budget の範囲で best root action を返す。`candidate_coverage.py` は policy ranking の後に構造的に必要な action type を補い、policy-only の blind spot を減らす。

Value は heuristic 実装に加えて `LinearValueModel` を利用できる。learned Value は `VALUE_FEATURE_SCHEMA_VERSION = 2` の feature を使い、artifact schema `3`、mask version `1.2`、学習時の exact `dto_version`、feature names を load 時に検証する。runtime inference は artifact の scaler/係数を使う線形評価で、terminal utility が DTO から exact に得られる場合はそれを優先する。

Policy は bootstrap の `PriorHeuristicPolicy` に加えて `LinearActionScorePolicy` を利用できる。learned `action_score` は Oracle root action の `estimated_q` の絶対値を回帰するのではなく、同一 decision 内の順位を pairwise logistic で distill する。runtime artifact は schema `1`、feature schema `4`、mask version `1.2`、exact `dto_version`、feature names を検証し、linear score を `ActionCandidate.action_score` に保持したまま候補を並べる。

Value の学習データは二系統を明確に分ける。supervised Value は Oracle の counterfactual `root_value_samples` を使い、actual trajectory loader は runtime で実際に commit された transition だけを読む。両者を混ぜて、実際の combat result を counterfactual action の label にすることはない。`action_score` 学習は pre-action DTO と `oracle_targets.root_actions[]` を使い、未解決 RNG outcome や `no_target` を numeric label に変換しない。

## 2. Architecture

| ファイル | 役割 |
|---|---|
| `combat_observation.py` | public Combat DTO から `CombatObservation` / `EnemyObservation` を作る。enemy block も保持する |
| `policy.py` | `PolicyModel` interface、`ActionCandidate`、`PriorHeuristicPolicy` |
| `action_score_features.py` | learned action ranking 用 feature schema v4。board/candidate/interaction feature を構築 |
| `action_score_training_data.py` | Oracle root-action label を読み、同一 decision 内 pairwise example を作る |
| `learned_policy.py` | artifact schema v1 の `LinearActionScorePolicy` と runtime contract validation |
| `value.py` | `ValueModel` interface と heuristic value |
| `value_features.py` | learned Value 用 feature schema v2 と DTO featurization |
| `learned_value.py` | artifact schema v3 の `LinearValueModel` と runtime contract validation |
| `value_training_data.py` / `_value_training_data_impl.py` | supervised root-value sample と actual committed trajectory の structured loader |
| `value_raw_data.py` | public Oracle JSONL を lossless に保持する raw loader |
| `beam_search.py` | branch/rng allocator、frontier expansion、stable scoring、trace、cleanup |
| `engine.py` | root decision orchestration、search/fallback/event branch の統合 |
| `combat_decision.py` | Combat action type ontology と continuation 判定 |
| `search_modes.py` | CLI/runtime 用の preset `BeamSearchConfig` |
| `candidate_coverage.py` | policy 後の structural coverage と provenance |
| `event_branch_search.py` | `choice_event_option` を branch simulation で比較 |

score 用語は 4 つに分ける。

| 用語 | 意味 |
|---|---|
| `state_score` | すでに stable/resolved/terminal になった 1 state に対する `ValueModel` 評価。`BeamNode.value` や `StablePruneNodeView.value` の canonical name。continuation node の inherited/stale value には使わない |
| `action_score` | simulation 前の候補 action に付いた policy score。bootstrap では heuristic、learned path では `LinearActionScorePolicy` の linear score。`ActionCandidate.action_score` に保持され、policy が score を出さない候補では `None` |
| `node_score` | search tree node 全体に帰属する score。`BeamSearchResult.best_value` では winning node の `state_score`、Oracle 集計では root action / RNG hypothesis の集約値も含む |
| `target_node_score` | supervised label として使う descendant-derived score。stable pruning target では、その node 自身の即時 value ではなく、後続 Oracle leaf/terminal から作られる |

`action_rank` / `policy_rank` と、RL の `behavior_frontier_scores` は ranking/sampling の情報であり、上の 4 つの score とは分けて読む。

Value feature schema v2 は HP/block/energy/enemy threat に加え、hand と draw/discard/exhaust pile の public card state を集約する。upgrade、enchantment、tinker-time、card type は feature に反映する一方、opaque な card id 自体は learned input にしない。raw DTO は再 featurization 用に data loader が保持する。

`action_score` feature schema v4 は同じ pre-action board に対して候補ごとの semantic feature を作る。raw board feature や choice operation/selection state の main effect は同一 decision の pairwise 差分では相殺されるため、board context × candidate semantics と choice context × candidate/card semantics の interaction を明示的に持つ。これにより danger/HP/energy や `gain` / `discard` / `exhaust` と、Attack/Skill/Potion/confirm/skip などの相対選好を linear pairwise model でも表現できる。opaque な action/card id は learned feature にしない。

`load_combat_value_rl_episodes()` は episode 内の `decision_index` だけでなく、隣接 step の `next_decision_point_id` と次 step の `decision_point_id`、前 step の `next_masked_emulator_dto` と次 step の `masked_emulator_dto` が一致することも検証する。`completed_only=True` でも filter 前に chain を検証する。

`load_combat_action_score_examples()` は既定で `exhaustive_root_actions=True` の Oracle record を要求する。`terminal` は weight `1.0`、`value_bootstrap` と fully resolved `mixed` は既定 `0.5`、`no_target` または 1 つでも未解決 RNG outcome を含む action は除外する。`build_pairwise_action_score_examples()` は Q の大小から winner/loser の feature delta とその符号反転を作り、対称な binary ranking examples にする。

## 3. API

```python
@dataclass
class BeamSearchConfig:
    beam_width: int = 8
    top_k_actions: int = 4
    max_depth: int = 2
    simulation_options: Mapping[str, Any] | None = None
    time_budget_ms: float | None = None
    max_batch_size: int = 64
    expand_partial: bool = True
    release_branches_on_finish: bool = True
    beam_searchable_action_types: frozenset[str] = field(default_factory=lambda: frozenset({"system", "card", "potion"}))
    max_continuation_steps: int = 8

class BeamSearchEngine:
    def __init__(self, client: Any, *, policy: PolicyModel, value_fn: ValueModel, config: BeamSearchConfig | None = None, stable_pruner: StableFrontierPruner | None = None, trace_collector: SearchTraceCollector | None = None) -> None
    async def search(self, instance_id: str, root_decision: Mapping[str, Any], *, timeout_s: float) -> BeamSearchResult
```

```python
class PolicyModel:
    def propose(self, legal_actions: Sequence[Mapping[str, Any]], masked_emulator_dto: Mapping[str, Any], *, top_k: int) -> list[ActionCandidate]
    def propose_batch(self, requests: Sequence[tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]], *, top_k: int) -> list[list[ActionCandidate]]
    def oracle_provenance(self) -> Mapping[str, Any]

class LinearActionScorePolicy(PolicyModel):
    @classmethod
    def from_weights_file(cls, path: str | Path) -> "LinearActionScorePolicy"
    def score_action(self, action: Mapping[str, Any], masked_emulator_dto: Mapping[str, Any]) -> float
    def propose(self, legal_actions: Sequence[Mapping[str, Any]], masked_emulator_dto: Mapping[str, Any], *, top_k: int) -> list[ActionCandidate]
    def oracle_provenance(self) -> Mapping[str, Any]
```

```python
class ValueModel:
    def evaluate(self, masked_emulator_dto: Mapping[str, Any]) -> float
    def evaluate_batch(self, dtos: Sequence[Mapping[str, Any]]) -> list[float]
    def exact_terminal_utility(self, masked_emulator_dto: Mapping[str, Any]) -> float | None
    def oracle_provenance(self) -> Mapping[str, Any]

class LinearValueModel(ValueModel):
    @classmethod
    def from_weights_file(cls, path: str | Path) -> "LinearValueModel"
    def evaluate(self, masked_emulator_dto: Mapping[str, Any]) -> float
    def exact_terminal_utility(self, masked_emulator_dto: Mapping[str, Any]) -> float | None
    def oracle_provenance(self) -> Mapping[str, Any]
```

Value training data:

```python
inspect_oracle_value_dto_contract(paths) -> OracleValueDtoContract
load_combat_value_examples(
    paths,
    *,
    terminal_weight=1.0,
    bootstrap_weight=0.5,
    allow_mixed_teachers=False,
) -> tuple[list[CombatValueTrainingExample], CombatValueDatasetStats]
load_combat_value_rl_episodes(paths, *, completed_only=False) -> list[CombatValueRLEpisode]
load_oracle_value_raw_records(paths) -> tuple[list[RawOracleValueRecord], OracleValueDtoContract]
load_raw_combat_value_episodes(paths) -> list[RawCombatValueEpisode]
```

Action-score training data:

```python
load_combat_action_score_examples(
    paths,
    *,
    terminal_weight=1.0,
    bootstrap_weight=0.5,
    mixed_weight=0.5,
    allow_mixed_teachers=False,
    require_exhaustive_root_actions=True,
) -> tuple[list[CombatActionScoreTrainingExample], CombatActionScoreDatasetStats]

build_pairwise_action_score_examples(
    examples,
    *,
    tie_tolerance=1e-9,
) -> list[PairwiseActionScoreExample]
```

```python
class CombatDecisionEngine:
    async def decide(self, instance_id: str, *, timeout_s: float, decision: Mapping[str, Any] | None = None) -> DecisionOutcome
    async def decide_and_commit(self, instance_id: str, *, timeout_s: float, decision: Mapping[str, Any] | None = None) -> dict[str, Any]
```

補助 API:

```python
available_action_types(dto) -> set[str] | None
is_continuation_decision(dto) -> bool
action_type_for_id(dto, action_id) -> str | None
apply_structural_coverage(candidates, legal_actions, *, top_k) -> list[ActionCandidate]
best_event_option(client, *, instance_id: str, decision_point_id: str, legal_actions: Sequence[JsonObject], timeout_s: float) -> str | None
```

## 4. 使用例

通常の decision:

```python
import asyncio

from sts2_training.api import AsyncTrainingApiClient, TcpConnection
from sts2_training.decision import BeamSearchConfig, CombatDecisionEngine


async def choose_once(instance_id: str) -> None:
    connection = TcpConnection(host="127.0.0.1", port=8765)
    async with connection:
        client = AsyncTrainingApiClient(connection)
        decision = await client.get_decision(instance_id, "root", timeout_s=30.0)
        engine = CombatDecisionEngine(
            client,
            beam_config=BeamSearchConfig(beam_width=8, top_k_actions=4, max_depth=3),
        )
        outcome = await engine.decide(instance_id, timeout_s=10.0, decision=decision)
        await client.commit_action(
            instance_id,
            decision["decision_point_id"],
            outcome.chosen_action_id,
            timeout_s=30.0,
        )


asyncio.run(choose_once("existing-instance-id"))
```

learned Value の学習:

```bash
python tools/train_combat_value.py \
  --log-dir data/combat_oracle \
  --output tools/output/combat_value_weights.json
```

learned `action_score` の学習:

```bash
python tools/train_combat_action_score.py \
  --log-dir data/combat_oracle \
  --output tools/output/combat_action_score_weights.json
```

runtime load:

```python
from sts2_training.decision.learned_policy import LinearActionScorePolicy
from sts2_training.decision.learned_value import LinearValueModel

policy = LinearActionScorePolicy.from_weights_file(
    "tools/output/combat_action_score_weights.json"
)
value_model = LinearValueModel.from_weights_file(
    "tools/output/combat_value_weights.json"
)
```

actual trajectory を読む場合:

```python
from sts2_training.decision.value_training_data import load_combat_value_rl_episodes

episodes = load_combat_value_rl_episodes(["data/combat_oracle/example.jsonl"])
terminal_return_episodes = load_combat_value_rl_episodes(
    ["data/combat_oracle/example.jsonl"],
    completed_only=True,
)
```

## 5. 補足説明

### 5.1 `beam_searchable_action_types` は runner が広げる

`BeamSearchConfig.beam_searchable_action_types` の既定値は `{"system", "card", "potion"}` であり、continuation（`choice_target` / `choice_card` / `choice_confirm` / `choice_skip`）を含まない。`resolve_search_mode()` が返す preset も同じ既定値を持つ。一方 `CombatDecisionEngine` は、明示的に渡された `beam_config` の `beam_searchable_action_types` を authoritative として尊重する。

したがって runner が preset をそのまま engine に渡すと、scope は狭いままになる。Emulator は `TargetType.AnyEnemy` のカードに対して**生存敵が 2 体以上のときだけ** `choice_target` continuation を発行するため、狭い scope は「複数敵戦でのみ、対象指定カードだけが探索から消える」という形で現れる。fault ではないので `branches_faulted` は 0 のままである。

runner が named/default preset から engine を作るときは `runner/beam_scope.py` の `runner_combat_beam_config()` を通す。widening を 1 箇所に集約するためのモジュールであり、entry point ごとに scope がずれないよう `tests/runner/test_runner_beam_scope.py` が entry point を pin している。呼び出し側が `BeamSearchConfig` を明示した場合は、その semantic scope を caller-authoritative として保持する。

scope 外で捨てられた branch は `BeamSearchStats.branches_out_of_scope` と `OutOfScopeDropTrace`（`event_type="out_of_scope_drop"`、`boundary` / `observed_action_types` / `allowed_action_types` 付き）に記録され、WARNING ログも出る。`branches_faulted` とは意図的に別カウンタで、非ゼロは transport/emulator の失敗ではなく **設定の誤り**を意味する。

### 5.2 その他

Beam Search は continuation handling、policy candidate limit、Whole Run active branch capacity、stable frontier pruning を別々の責務として扱う。`StableFrontierPruner` が制御するのは stable/resolved frontier の survivor selection だけであり、詳細は [03_stable_pruner.md](03_stable_pruner.md) を参照する。

Value supervised data の `root_value_samples` は Oracle が生成した counterfactual post-state/target であり、actual committed trajectory とは別物である。`action_score` の `estimated_q` も Oracle teacher が同一 root decision 内の ranking label として生成した値で、runtime linear score を calibrated Q と解釈しない。target の生成規則、Oracle record schema、teacher/search-budget provenance は [04_oracle.md](04_oracle.md) を参照する。structured dataclass で未使用の public producer field まで保持したい場合は `value_raw_data.py` の raw loader を使う。
