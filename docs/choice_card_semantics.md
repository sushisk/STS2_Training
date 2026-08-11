# Canonical choice-card semantics in Training

Training consumes the public `masked_emulator_dto.pendingChoice` contract produced by
STS2_Emulator and normalized/masked by STS2_RL. It does not infer card-choice mechanics
from prompt text, selector names, labels, card IDs, or incidental option shape.

This is the downstream companion to:

- STS2_RL #34 (`choiceSemantics` v1 public boundary and state-key identity)
- STS2_Emulator #3 (producer-side mechanic descriptors and decision-local option IDs)

## Consumer rule

Use `sts2_training.selection.pending_choice_context(masked_emulator_dto)` to read:

- canonical `choiceSemantics` v1 (`operation`, effect/zone/modifier factors)
- opaque `sourceEffectId` when semantics are known
- `selectedOptionIds`
- remaining option IDs

Use `sts2_training.selection.choice_option_id(action)` to read the opaque `optionId`
attached to a `choice_card` legal action.

Malformed, absent, or future semantic descriptors degrade to
`operation="unknown"`. Training must stay policy-neutral in that case.

## Rollout

The transport `schema_version` remains `0.7`; RL #34 changes the mask contract to 1.1.
Older Emulator builds therefore remain consumable: pending card choices simply appear
with neutral/unknown semantics after RL normalization. Training's accessor follows the
same rule so rollout does not require heuristic reconstruction of mechanic meaning.
