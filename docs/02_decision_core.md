# decision core

## 0. 文章の目的

この文書は `decision/` の中核である Beam Search、Policy、Value、CombatDecisionEngine、Combat DTO 正規化、event branch search、candidate coverage を説明する。stable frontier pruning の学習部分は [03_stable_pruner.md](03_stable_pruner.md)、Oracle は [04_oracle.md](04_oracle.md) に分ける。

## 1. 概要

decision core は、1 つの Combat decision DTO から action を選ぶ層である。`CombatDecisionEngine` はまず event choice の特殊処理を試し、Combat search 可能な状態なら `BeamSearchEngine` を実行し、actionable な search result がない場合は fallback selector に戻す。

`BeamSearchEngine` は `PolicyModel` が提案した候補 action を `AsyncTrainingApiClient.emulate_action(s)` で分岐実行し、`ValueModel` が stable/terminal state を評価し、beam width と depth budget の範囲で best root action を返す。`candidate_coverage.py` は policy ranking の後に構造的に必要な action type を補い、policy-only の blind spot を減らす。

## 2. Architecture

| ファイル | 役割 |
|---|---|
| `combat_observation.py` | public Combat DTO から `CombatObservation` / `EnemyObservation` を作る |
| `policy.py` | `PolicyModel` interface、`ActionCandidate`、`PriorHeuristicPolicy` |
| `value.py` | `ValueModel` interface と heuristic value |
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
| `action_score` | simulation 前の候補 action に付いた policy score。`policy_score` の canonical name。policy が score を出さない候補では `None` |
| `node_score` | search tree node 全体に帰属する score。`BeamSearchResult.best_value` では winning node の `state_score`、Oracle 集計では root action / RNG hypothesis の集約値も含む |
| `target_node_score` | supervised label として使う descendant-derived score。stable pruning target では、その node 自身の即時 value ではなく、後続 Oracle leaf/terminal から作られる |

`action_rank` / `policy_rank` と、RL の `behavior_frontier_scores` は ranking/sampling の情報であり、上の 4 つの score とは分けて読む。

## 3. API

```python
@dataclass
class BeamSearchConfig:
    beam_width: int = ...
    top_k_actions: int = ...
    max_depth: int = ...
    timeout_s: float | None = ...

class BeamSearchEngine:
    def __init__(self, client, policy: PolicyModel | None = None, value_fn: ValueModel | None = None, *, config: BeamSearchConfig | None = None, stable_pruner: StableFrontierPruner | None = None, trace_collector: SearchTraceCollector | None = None)
    async def search(self, decision: Mapping[str, Any], *, deadline: float | None = None) -> BeamSearchResult
```

```python
class PolicyModel:
    def propose(self, dto: Mapping[str, Any], *, top_k: int) -> list[ActionCandidate]
    def propose_batch(self, dtos: Sequence[Mapping[str, Any]], *, top_k: int) -> list[list[ActionCandidate]]
    def oracle_provenance(self) -> Mapping[str, Any]

class ValueModel:
    def evaluate(self, dtos: Sequence[Mapping[str, Any]]) -> list[float]
    def exact_terminal_utility(self, dto: Mapping[str, Any]) -> float | None
    def oracle_provenance(self) -> Mapping[str, Any]
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
best_event_option(client, decision, *, timeout_s: float = ...) -> Mapping[str, Any] | None
```

## 4. 使用例

```python
import asyncio

from sts2_training.api import AsyncTrainingApiClient
from sts2_training.decision import BeamSearchConfig, CombatDecisionEngine


async def choose_once(instance_id: str) -> None:
    async with AsyncTrainingApiClient() as client:
        decision = await client.get_decision(instance_id, "root")
        engine = CombatDecisionEngine(
            client,
            beam_config=BeamSearchConfig(beam_width=8, top_k_actions=4, max_depth=3),
        )
        outcome = await engine.decide(instance_id, timeout_s=10.0, decision=decision)
        await client.commit_action(
            instance_id,
            decision["decision_point_id"],
            outcome.chosen_action_id,
        )


asyncio.run(choose_once("existing-instance-id"))
```

## 5. 補足説明

Beam Search は continuation handling、policy candidate limit、Whole Run active branch capacity、stable frontier pruning を別々の責務として扱う。`StableFrontierPruner` が制御するのは stable/resolved frontier の survivor selection だけであり、詳細は [03_stable_pruner.md](03_stable_pruner.md) を参照する。Oracle collection は runtime engine とは別 teacher copy を使うため、[04_oracle.md](04_oracle.md) で扱う。
