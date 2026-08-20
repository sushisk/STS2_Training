# STS2 RL / Training wire contract v0.8

This document is the canonical compatibility delta for the `0.8` RL/Training API wire format.
The transport, session sequencing, request identity, operation set, branch lifecycle, retry,
and at-most-once rules remain those documented for v0.7 unless overridden below.

## Deployment model

DTO v0.8 is a deliberate **hard cutover** from v0.7. RL and Training must be deployed or
activated as one lockstep compatibility unit. A v0.8 Training client must not be routed to
a v0.7 RL endpoint, and a v0.7 Training client must not be routed to a v0.8-only endpoint.
Both sides therefore advertise/accept only `schema_version = "0.8"` after this cutover.

This version axis is the **Training↔RL wire DTO version**. It is separate from the
CombatStateSnapshot/Restore contract version used inside RL/Emulator integration, even when
both happen to use the number `0.8`.

## Breaking change: masked pile multiset representation

Wire DTO v0.8 pairs with `masked_emulator_dto.mask_version = "1.2"`.

In v0.7-era masking, `drawPile`, `discardPile`, and `exhaustPile` were represented as a
`{card_id: count}` JSON object. That representation collapsed distinct card instances that
shared an id but differed in upgrade level, Tinker Time state, or Enchantment.

In v0.8 the three masked piles are arrays of per-distinct-card-identity records. Each record
contains the public card fields plus `count`, including:

- `id`
- `type`, `rarity`, `cost`, `targetType`
- `upgraded`, `upgradeLevel`
- `tinkerTimeType`, `tinkerTimeRider`
- `enchantment` (`id`, `amount`, `status`) when present
- `count`

The identity grouping key is `(id, upgradeLevel, tinkerTimeType, tinkerTimeRider,
enchantment.id, enchantment.amount, enchantment.status)`. Pile order remains hidden; card
instance state does not.

Because the JSON type changes from object to array, this is a breaking wire change and is
the reason the outer wire `schema_version` advances from 0.7 to 0.8 rather than relying on
`mask_version` alone.

## Card-instance fidelity

The paired v0.8 rollout also preserves `upgradeLevel` and `enchantment` when reconstructing
CombatScenario card instances from masked state, including pending-choice card options.
Scenario schema validation requires an Enchantment `id` whenever an enchantment object is
present.

For the currently supported game build, negative `upgrade_level` values are rejected because
scenario restoration only exposes forward upgrades through repeated `CardCmd.Upgrade` calls.
This is a current-build capability constraint, not a claim that future game versions can never
give negative levels a meaning; revisit the bound if the game contract changes.

## Optional fault diagnostics on `emulate_actions` branch results

A per-Branch `faulted` result may carry an optional `diagnostics` object. It is advisory:
Training must treat its absence as normal and must not depend on any key inside it. The
terminal per-Branch fields (`status`, `branch_id`, `parent_branch_id`, `rng_id`, `error`,
`fault_kind`) keep the meaning documented for v0.7.

```json
{
  "status": "faulted",
  "branch_id": "bs-0001",
  "parent_branch_id": "root",
  "rng_id": 7,
  "error": "boundary mismatch",
  "fault_kind": "replay_mismatch",
  "diagnostics": {
    "fault_kind": "replay_mismatch",
    "message": "boundary mismatch",
    "expected_boundary": "stable",
    "actual_boundary": "pending_choice",
    "actual_choice_scope": "TopLevel",
    "actual_choice_kind": "target",
    "actual_room_context": { "room_type": "CombatRoom" },
    "actual_masked_emulator_dto": { "...": "..." }
  }
}
```

`diagnostics` describes **where the failed Branch actually landed**, which is what makes a
replay mismatch diagnosable at all: the generic `error` string cannot say what state was
reached instead of the expected one.

Everything in `diagnostics` is subject to the same masking as any other published state.
`actual_masked_emulator_dto`, when present, is built by the same masking path as a normal
decision payload and carries the same `dto_version` and `mask_version` stamps; the raw
Emulator observation is never published. `actual_boundary`, `actual_choice_scope`,
`actual_choice_kind`, and `actual_room_context` correspond to `boundary`, `choiceScope`,
`pendingChoice.choiceType`, and `room_context` of that masked DTO.

RL retains the unmasked detail in its own server log, keyed by `branch_id`, so an operator
can correlate a Training-side fault record with the full server-side context without that
context crossing the wire.

Adding `diagnostics` is backward compatible in both directions: Training validates the
required per-Branch fields and ignores unknown keys, and an RL endpoint that omits the
field remains conformant.

## Compatibility gate

The three paired PRs are intended to merge and deploy together:

- STS2_Emulator #7: card upgrade/enchantment scenario fidelity and public card DTO fields.
- STS2_RL #41: mask_version 1.2 pile records and lossless scenario reconstruction.
- STS2_Training #57: scenario harvesting from mask_version 1.2 logs and wire DTO v0.8 client.

Acceptance requires the paired Training client and RL endpoint to exchange
`schema_version = "0.8"`, and harvested v0.8 masked states to reconstruct card upgrade and
enchantment state without silently collapsing pile entries.
