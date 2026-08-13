# Combat ValueModel learning

Combat Value data comes from Oracle v6 JSONL. Collection is the expensive teacher/data
step; fitting and future reinforcement-learning consumers are offline.

The base-data rule is **preserve public information by default**. Raw
`masked_emulator_dto` and public response/action/runtime metadata are kept; hidden
Emulator state that the RL masking contract intentionally removes is never reintroduced.
Full deep-Beam DTOs remain the deliberate data-volume exception: only root-action
post-state DTOs are persisted for counterfactual Value supervision.

## 0. Terminology

Four distinct scores appear throughout Oracle collection and this doc. They differ along
two axes: what they score (a candidate action vs. a resolved board state vs. a search-tree
node) and when they're computed (before vs. after simulating an action). Getting these
confused is easy, so names are fixed here:

- **`action_score`** — scores a candidate action *before* its result is known (cheap, no
  simulation required). Used to shortlist which legal actions are worth simulating at all.
  Current implementation: `PolicyModel`/`PriorHeuristicPolicy` (`decision/policy.py`),
  exposed today as `policy_score`/`policy_rank`. Not yet learned.
- **`state_score`** — scores one resolved (simulated) board state in isolation, with no
  reference to how it was reached. Current implementation: `ValueModel.evaluate()`
  (`decision/value.py`); this is the function PR #59 trains from Oracle v6 data.
- **`node_score`** — the value of one specific search-tree node/branch: the `state_score`
  of the state that branch's action leads to, attributed to that branch. Not an
  independently learned function — it *is* `state_score`, applied post-simulation and
  credited to the action that produced it. Logged today as `estimated_q` in
  `oracle_targets.root_actions[]`. Matches this codebase's own `BeamNode` (one candidate
  inside a beam, as opposed to `beam_width`, the size of the whole candidate set).
- **`frontier_score`** — given a node's `state_score` and its context within the current
  search frontier (rank, gap to the best sibling, depth, remaining beam budget), decides
  whether that node is worth continuing to expand. Current implementation:
  `stable_pruner.py`/`learned_pruner.py` (PR #50) — already learned, not part of the
  #58/#59 scope.

Collection order in one Oracle search step:

```text
action_score  (cheap, pre-simulation shortlist of candidates)
    -> simulate each shortlisted candidate
    -> state_score  (score the resulting board)
    -> node_score   (that state_score, attributed to the branch/action - "estimated_q")
    -> frontier_score  (decide which nodes survive to the next search depth)
```

`exhaustive_root_actions` (default `True`) bypasses `action_score` filtering at the root
specifically so every legal action gets a real, unbiased `node_score` from Oracle search -
this is what makes root_value_samples/root_actions usable as `action_score` training
labels later without inheriting today's heuristic's blind spots.

## 1. Collect Oracle v6 data

Collection requires the hard-cutover Training/RL wire contract and masked DTO v1.2. The
record also pins the exact public `dto_version`, because two Emulator generations may
share mask v1.2 while still giving state/value labels different semantics.

```bash
python -m sts2_training.runner.oracle_collection \
  --scenario data/scenarios/example.json \
  --output data/combat_oracle/example-001.jsonl \
  --oracle-beam-width 32 \
  --oracle-top-k 8 \
  --oracle-depth 4
```

Each `combat_oracle_decision` contains both teacher counterfactuals and the actual runtime
transition:

```text
public root decision DTO + public response metadata
    ├── oracle_targets.root_actions      # Policy/top-k teacher
    ├── oracle_targets.stable_nodes      # Stable-pruner teacher
    ├── root_value_samples               # root post-state + deeper Oracle Value target
    └── runtime_transition               # action actually committed + public next DTO
```

The record also carries `instance_id`, zero-based `decision_index`, and a `dto_contract`
containing wire schema, mask version, and `dto_version`. `root_value_samples` retain the
full public root action payload, RNG id, post-state DTO, target source/value, deepest
combat depth, and censoring/best-node metadata.

Both Oracle and runtime Beam outcome diagnostics are preserved as bounded summaries. They
keep useful best action/value/reason/stats and best-node identity/ranking/action metadata,
but explicitly omit a deep best-node `masked_emulator_dto` and `branch_log`. This closes
the two paths that could otherwise reintroduce speculative deep-Branch payloads while
preserving the root-only DTO volume boundary.

`runtime_transition` is intentionally separate from the Oracle samples. It records the
actual chosen action payload, runtime decision source, bounded Beam diagnostic summary,
commit-response public metadata, next decision id, raw next masked DTO, and per-step
combat result when terminal.

The final `combat_oracle_episode_result` record stores completion/truncation status,
normalized victory/defeat when known, final public response metadata, exact DTO contract,
and the final raw masked DTO. Failed collection episodes are rolled back atomically rather
than leaving partial training data.

### Record field reference

Exhaustive field list for both JSONL record types. The rule stated above ("preserve public
information by default") means every field below is kept verbatim except the one marked
`[bounded]`; there is exactly one deliberate omission in the whole schema.

**`combat_oracle_decision` (one per root Decision):**

- `record_type`, `record_schema_version` (currently 6), `instance_id`, `decision_index`,
  `decision_point_id`
- `dto_contract`: `wire_schema_version`, `mask_version` (fixed at `1.2`), `dto_version` —
  every DTO elsewhere in this record must match this exactly
- `decision_response_metadata`: every public response-envelope field of the root decision
  except `masked_emulator_dto` itself (kept separately below, to avoid duplicating the
  largest field)
- `masked_emulator_dto`: the root decision's raw DTO, unmodified
- `root_value_samples[]` — one root-action post-state each:
  - `action_id`, `action` (full payload), `rng_id`, `root_state_node_id`
  - `decision_point_id` — the *post-state's* next decision id; `None` for a terminal
    post-state (there is no next decision), never an empty string
  - `masked_emulator_dto`: that post-state's raw DTO
  - `target_value`, `target_source` (`terminal` / `value_bootstrap` / `no_target`),
    `terminal_reached`, `deepest_combat_depth`, `censored`, `censor_reason`, `best_node_id`
- `oracle_search_result` **[bounded]**: `best_root_action_id`, `best_value`, `reason`,
  `stats` are kept in full; `best_node` keeps `branch_id`, `parent_branch_id`, `rng_id`,
  `decision_point_id`, `depth`, `value`, `root_action_id`, `combat_depth`,
  `continuation_steps`, `terminal`, `action_id`, `action_type`, `action`, `policy_rank`,
  `policy_score`, `post_coverage_rank`, `candidate_source` — but explicitly omits
  `masked_emulator_dto` and `branch_log` (listed in `omitted_large_fields`). This is the
  **only** omission anywhere in the schema; it exists purely because branch-tree DTO volume
  scales with search depth/width, not because the information is hidden by policy.
- `oracle_targets`:
  - `metadata`: `search_id`, `oracle_beam_width`, `target_beam_width`, `top_k_actions`,
    `max_depth`, `max_continuation_steps`, `time_budget_ms`, `exhaustive_root_actions`,
    `rng_sampling`, `search_reason`, `pruner_name`, `pruner_version` — these plus
    `rng_sampling` are exactly the fields `oracle_teacher_provenance.py` fingerprints as
    "target generation config", so mixing incompatible search budgets/pruners across a
    dataset is rejected rather than silently pooled
  - `root_actions[]`: `action_id`, `action`, `evaluated`, `estimated_q`, `rng_outcomes[]`
    (each: `rng_id`, `value`, `target_source`, `terminal_reached`, `deepest_combat_depth`,
    `censored`, `censor_reason`, `best_node_id`), `target_source`, `terminal_reached`,
    `censored`, `censor_reason`
  - `stable_nodes[]`: `prune_step_id`, `node_id`, `root_action_id`,
    `frontier_index_before_prune`, `oracle_kept`, `target_beam_width`,
    `baseline_would_keep`, `target_value`, `target_source`, `terminal_reached`, `censored`,
    `censor_reason`, `best_descendant_node_id`
- `search_trace[]` — one entry per resolved search-tree node: `search_id`, `node_id`,
  `parent_node_id`, `branch_id`, `parent_branch_id`, `root_action_id`, `rng_id`,
  `decision_point_id`, `depth`, `combat_depth`, `continuation_steps`, `value`,
  `value_is_fresh`, `value_source` (`terminal` / `value_bootstrap` / `inherited`),
  `state_kind`, `resolution`, `terminal`, `action_id`, `action_type`, `action`,
  `policy_rank`, `policy_score`, `post_coverage_rank` — no DTO field exists here at all,
  which is the trace-level side of the same deep-Branch-DTO exclusion as
  `oracle_search_result.best_node`
- `runtime_transition` — the action actually committed, distinct from the counterfactual
  Oracle samples above: `chosen_action_id`, `chosen_action` (full payload),
  `decision_source`, `beam_result` **[bounded, same shape/omission as
  `oracle_search_result`]**, `next_decision_point_id`, `commit_response_metadata`,
  `next_masked_emulator_dto` (raw post-commit DTO), `next_dto_contract`, `combat_result`
- `provenance`: `training_commit`, `teacher_policy_class`, `teacher_inner_policy_class`,
  `teacher_coverage_policy_class`, `teacher_value_class`, `teacher_policy_metadata`,
  `teacher_inner_policy_metadata`, `teacher_value_metadata`, `pruner_name`,
  `pruner_version`, `rng_sampling`

**`combat_oracle_episode_result` (one per episode, appended last):**

- `record_type`, `record_schema_version` (currently 2), `instance_id`,
  `decisions_collected`, `completed`, `termination_reason`, `combat_result` (normalized
  `victory`/`defeat`/`None`), `dto_contract`, `final_decision_metadata`,
  `final_masked_emulator_dto` (raw final DTO), `elapsed_s`
- Failed episodes never reach this record: on any exception the writer rolls the JSONL
  output back to its pre-episode size, so partial episodes (decision records with no
  matching episode-result) cannot appear rather than being tagged as errors after the fact.

## 2. Supervised Oracle Value training

Install training dependencies and fit the current weighted ridge model.

```bash
pip install -e ".[train]"
python tools/train_combat_value.py \
  --log-dir data/combat_oracle \
  --output tools/output/combat_value_weights.json
```

Supervised fitting uses **only `root_value_samples`**. `terminal` targets receive weight
1.0, `value_bootstrap` targets 0.5, and `no_target` samples remain censored/unknown.
Counterfactual root samples are never labeled with the result of the actual committed
trajectory.

Value feature schema v2 includes HP/block/energy/enemy threat plus public card-state
aggregates across hand and draw/discard/exhaust multisets. Pile `count` is multiplicity;
upgrade level, enchantment, tinker-time state, and Attack/Skill/Power semantics affect the
feature vector. Opaque card ids are not model inputs, while the raw DTO remains available
for later re-featurization.

The dataset loader requires one exact `(wire schema, mask version, dto_version)` across the
input dataset. Learned artifact schema v3 writes that exact generation directly as
`required_dto_version` at the artifact top level, while retaining the same value in split
metrics for auditability. The artifact also contains feature schema/hash, input
hashes/splits, teacher/search provenance, scaler, coefficients, terminal utility metadata,
and regression metrics.

## 3. Actual trajectory / future Value RL data

`load_combat_value_rl_episodes()` is a structured loader for **actual committed** data. It
groups decision records by public episode identity, checks contiguous `decision_index`,
requires the matching episode-result record, verifies that the final DTO equals the final
logged transition, and returns the public pre-state/action/post-state fields needed by
current trajectory consumers.

It deliberately does not choose an RL objective or manufacture rewards. Completed
victory/defeat episodes expose `usable_for_terminal_return=True`; deliberately truncated
collections are retained by default for TD/bootstrap or audit use and can be excluded with
`completed_only=True`.

```python
from sts2_training.decision.value_training_data import load_combat_value_rl_episodes

episodes = load_combat_value_rl_episodes(["data/combat_oracle/example-001.jsonl"])
terminal_return_episodes = load_combat_value_rl_episodes(
    ["data/combat_oracle/example-001.jsonl"],
    completed_only=True,
)
```

This boundary prevents the critical error of assigning the actual combat result to every
counterfactual `root_value_samples[]` branch.

### Lossless foundation loader

Structured dataclasses inevitably choose fields. To keep the foundation producer-favored,
`value_raw_data.py` provides a parallel lossless path:

```python
from sts2_training.decision.value_raw_data import (
    load_oracle_value_raw_records,
    load_raw_combat_value_episodes,
)

records, dto_contract = load_oracle_value_raw_records(paths)
episodes = load_raw_combat_value_episodes(paths)
```

`RawOracleValueRecord.payload` is a deep copy of the complete public JSONL object. Known
v6/v2 records and exact DTO contracts are validated, but unknown record types and
producer-added public fields are retained instead of silently projected away. Raw episode
grouping likewise stores the complete decision records and complete episode-result record.
This is the preferred foundation seam when designing a future RL schema or feature set.

## 4. Held-out evaluation

```bash
python tools/eval_combat_value.py \
  --weights tools/output/combat_value_weights.json \
  --log-dir data/combat_oracle_heldout
```

Teacher/search provenance and exact DTO generation must match by default. Reports include
label coverage, MAE/RMSE, weighted MAE/RMSE, R-squared when defined, and upgrade/enchantment
coverage.

## 5. Runtime use

Runtime inference has no numpy/scikit-learn dependency.

```python
from sts2_training.decision.engine import CombatDecisionEngine
from sts2_training.decision.learned_value import LinearValueModel

value_model = LinearValueModel.from_weights_file(
    "tools/output/combat_value_weights.json"
)
engine = CombatDecisionEngine(client, value_fn=value_model)
```

Runtime checks mask v1.2 **and the exact artifact-pinned `dto_version` before terminal
short-circuiting or learned scoring**. A model is therefore not silently applied to a new
Emulator data generation merely because its mask version stayed unchanged.

## 6. Interpretation boundary

Oracle values are budget-dependent teacher estimates, not ground truth. Actual episode
victory/defeat is a different signal. Supervised Oracle distillation and future Value RL
must remain separate data paths even though one Oracle v6 collection file intentionally
contains enough public information to support both.

A green repo-local fit establishes contract/implementation consistency only. Promote a
learned ValueModel after held-out Oracle evaluation and fixed-seed real-Emulator A/B
validation.
