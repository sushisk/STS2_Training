# STS2_Training API connection

Training-side implementation for the `sushisk/STS2_RL` async TCP / DTO v0.7 contract.

The supported path is now deliberately **async + TCP only**. The legacy synchronous
`TrainingApiClient` / `LocalProcessTransport` path is retired for v0.7 because it cannot
participate in the session handshake and `server_epoch` safety model.

## TCP smoke test

Start RL separately, then run:

```bash
python -m sts2_training.api.tcp_smoke --host 127.0.0.1 --port 8765
```

A new TCP stream first sends an exact transport hello containing a stable
`client_session_id`. RL returns its `server_epoch`. The smoke test then sends ping and
prints a response similar to:

```json
{"server_epoch":"...","transport_operation":"pong"}
```

## Async DTO API client

```python
import asyncio

from sts2_training.api import AsyncTrainingApiClient, TcpConnection


async def main() -> None:
    connection = TcpConnection(host="127.0.0.1", port=8765)
    async with AsyncTrainingApiClient(connection) as client:
        instance_id = await client.start_instance(
            {"instance_type": "combat"},
            timeout_s=30.0,
        )
        decision = await client.get_decision(instance_id, timeout_s=30.0)
        print(decision)


asyncio.run(main())
```

## Session sequencing

Each client owns one `client_session_id` and sends strictly increasing `request_seq`
values. `request_id` is deterministic: `<client_session_id>:<request_seq>`.

The client advances its sequence only after receiving a definitive API response. If a
request may have reached RL but no valid response was observed, the exact serialized DTO
is exposed as `client.pending_retry` and all fresh operations fail closed.

Recovery is explicit:

```python
pending = client.pending_retry
if pending is not None:
    result = await client.retry_request(pending, timeout_s=30.0)
```

RL keeps the most recent executable request/response for every logical session. Exact
same-sequence retry is therefore replayed rather than executed again. A different payload
with the same sequence or a sequence gap is rejected.

## RL restart semantics

Every API response and transport hello/pong contains `server_epoch`. A reconnect must see
the same epoch. If RL restarted, `TcpConnection` raises `ServerEpochChangedError` and the
`AsyncTrainingApiClient` becomes permanently invalid. Create a new client/session; do not
retry the unresolved request into the new RL process.

This is intentional: v0.7 guarantees at-most-once execution within one RL process epoch,
not durable exactly-once execution across emulator process restarts.

## Timeouts, cancellation, and response limits

A public API `timeout_s` covers waiting for the client operation lock and the TCP
exchange. Cancellation/timeout after send invalidates the stream and preserves the exact
pending request.

`TcpConnection(max_response_bytes=...)` bounds response buffering independently from the
request frame limit. If a valid cached response is larger than the local receiver bound,
raise the bound with `await connection.set_max_response_bytes(...)` and replay the exact
`pending_retry`; changing the local receiver limit does not change request identity.

## Selection audit

`AsyncTrainingApiClient` still accepts a `SelectionEventLogger`. A completion-uncertain
selection is logged on the first attempt; replay of the same `request_id` is recorded as
selection recovery rather than a second logical selection.

## Board evaluation: Run-state value model

`sts2_training.board_eval` is the Whole Run state-value path. It is intentionally separate
from Combat tactical scoring: `RunStateValueModel` does **not** inherit
`decision.value.ValueModel`, and `LinearRunStateValueModel` is not passed to
`CombatDecisionEngine(value_fn=...)`.

The card/deck input pipeline reads `tools/output/card_secondary_features.csv` through a
repository-anchored default path. `CardFeatureExtractor` produces fixed-order
`CardFeatures`, and `summarize_deck()` produces `DeckSummary`. The former
`other_effect_magnitude` catch-all has been split into effect families such as heal, max-HP,
gold, card generation/transform/tutor, buff/debuff, orb, character-resource, known-other,
and unparsed counts. Upgrade handling is deliberately instance-count based only;
`upgraded_count` / `upgrade_ratio` and type-specific upgraded counts are the final
representation rather than guessed upgraded damage/block deltas.

Energy and Stars are separate resource dimensions. Star-cost cards keep `star_cost` and do
not enter the ordinary Energy cost bins as free 0-Energy cards. Upstream uncertain counts
such as `SOUL:1?` are kept separate from confirmed generation/transform/tutor/orb counts.
`damage_per_energy` and `block_per_energy` use only Energy-cost cards contributing the
corresponding effect.

Referential cards retain structured `ReferenceScaling` data with a separate `referent`
(`self` / `enemy` / `None`) and `filter_value`. Dynamic multi-hit cards additionally retain
`hit_count_reference`, recording what determines hit count without inventing a numeric hit
count or `effective_total_damage`. Deck summaries also expose `unknown_card_count` and
`known_card_ratio`; unknown cards may be skipped for live Run-state inference without
silently disappearing from model coverage features. No coverage-threshold fallback or
runtime catalog-version rejection is added.

The planned neural Deck Embedder remains a follow-up to the current engineered logistic
baseline. Its boundary is:

```text
Card ID Embedding
+ Card instance/mechanical features
        ↓
Shared Card MLP
        ↓
64-d Card Vector per card
        ↓
SUM / MEAN / MAX pooling
        ↓
192-d Deck Embedding
```

The per-card side is intended to carry card identity plus the mechanical/instance signals
represented by `CardFeatures`: upgrade state; Energy/Star cost; damage, block, draw and
energy gain; card type and AOE/multi-hit/keyword/scaling flags; separated effect-family and
uncertainty signals; and structured dynamic/reference metadata where applicable. Enchantment
will join the CardInstance state after its explicit follow-up implementation. `DeckSummary`
is **not** pooled into this embedding: it remains a parallel engineered input, alongside the
Deck Embedding and later Relic / Encounter / Floor / HP / Gold state features, to the final
value model. The initial Deck Encoder uses SUM / MEAN / MAX pooling; MIN is intentionally
not part of the initial design.

`board_eval.training_data.build_examples_from_log` turns one self-play JSONL Run into
Win/Lose-labeled examples. `hp`, `max_hp`, `gold`, `act_floor`, and `total_floor` each have a
matching `*_missing` feature so a missing value is distinguishable from a genuine zero.
For log-derived examples, `state_kind` uses `currentRoomType` first and then the selection
event's `boundary`; DTO-local `boundary` remains a final compatibility fallback. Callers can
optionally filter `build_examples_from_log(..., state_kinds=...)`, including values obtained
from the event-level boundary. The concrete Whole Run state-kind policy remains a follow-up
decision.

`tools/train_board_eval.py` fits a CPU-only `LogisticRegression` over
`MODEL_FEATURE_NAMES`. Train/validation/test are split by whole Run and stratified by final
Win/Lose outcome. The output JSON includes coefficients/scaling plus model/artifact schema,
creation time, card-catalog SHA-256/path, training commit, training state-kind scope, and
feature-schema hash metadata. These metadata are recorded for traceability;
`LinearRunStateValueModel.from_weights_file` does not enforce catalog/version matching.

```python
from sts2_training.board_eval import LinearRunStateValueModel

model = LinearRunStateValueModel.from_weights_file(
    "tools/output/board_eval_model_weights.json"
)
win_probability = model.predict_win_probability(masked_emulator_dto)
```

See `src/sts2_training/board_eval/how_to_use.md` for the full feature contract, training
options, and current limitations.

## Decision logic: Policy + Beam Search + Value function

`sts2_training.decision.CombatDecisionEngine` wires `get_decision` -> policy-guided
beam search (`emulate_actions` batching, one logical batch per beam depth, chunked to the
active instance's published capacity) -> value-function scoring -> `commit_action` on top
of `AsyncTrainingApiClient`. It falls back to `selection.HeuristicCombatSelector` when
beam search cannot safely branch (for example, a non-combat boundary or a rejected batch).
Unexpected policy/value implementation errors are surfaced instead of silently converted
into heuristic decisions. `PolicyModel`/`ValueModel` are abstract bases with runnable
heuristic defaults so the pipeline works end-to-end before any trained checkpoint exists -
see `src/sts2_training/decision/how_to_use.md` for the full usage guide, config knobs, and
how to plug in real models.

## Runner: top-level entry points

`sts2_training.runner` starts an instance and drives it to completion on top of
`CombatDecisionEngine`, via two entry points sharing one loop (`EpisodeRunner`):
`start_combat_from_state` (Combat from a fully-specified board, `CombatScenario`) and
`start_new_run` (a normal, from-scratch Whole Run, `NewRunConfig`). Each module doubles
as a CLI (`python -m sts2_training.runner.start_new_run --help`).
`search_mode`/`beam_max_depth` (or `--search-mode`/`--beam-depth` on the CLI) pick a
named beam search preset (see `sts2_training.decision.search_modes`) without hand-
constructing a `BeamSearchConfig`. See `src/sts2_training/runner/how_to_use.md` for
the full guide.

### Self-play data collection

`sts2_training.runner.run_self_play_batch` drives many `start_new_run` Whole Runs
concurrently (bounded by `concurrency`, one `TcpConnection` per run) and logs every
selection Training makes - the committed action and every beam-search-explored-but-
unchosen branch - to its own JSONL file via `selection_log.JsonlSelectionLogger`. It
adds no new decision logic; the policy is still `CombatDecisionEngine`'s heuristic
default, deliberately random for non-combat boundaries (map/shop/rest/reward) per
`HeuristicCombatSelector`'s own "initial data-collection stage" placeholder design.
This is a bootstrap data source for a future board/deck evaluation model (Run-level
Win/Lose labels). One run failing is captured per-run rather than aborting the batch.
CLI:

```sh
python -m sts2_training.runner.self_play \
    --host 127.0.0.1 --port 8765 --character-id IRONCLAD \
    --num-runs 50 --concurrency 8 --output-dir data/self_play
```
