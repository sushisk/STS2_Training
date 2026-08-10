# Whole Run action_id paired cutover

This change is intentionally paired with `sushisk/STS2_RL#10`.

## Why the pair must move together

Training PR #28 contains a temporary compatibility workaround for the pre-#10 Whole Run server: selection still uses the public sparse `action_id`, but root `commit_action` translates that ID to its ordinal position before sending it on the wire.

RL #10 fixes the server contract instead. Whole Run resolves the wire token by exact `action_id` equality against the current legal actions, matching the DTO rule that Training returns the published ID unchanged.

The two fixes are not independently deployable:

| Training | RL | Result |
| --- | --- | --- |
| #28 compatibility behavior | pre-#10 | Supported temporary pair |
| #28 compatibility behavior | #10 | **Unsupported**: ordinal may be rejected or select a different real ActionId |
| this PR | pre-#10 | **Unsupported**: sparse public ID is still interpreted as a positional index |
| this PR | #10 | Supported contract-correct pair |

For example, a Reward Decision with public IDs `["0", "3"]` illustrates both failure directions:

- #28 + pre-#10: selecting public `"3"` sends ordinal `"1"`; old RL indexes slot 1 and executes ActionId `3`.
- #28 + RL #10: selecting public `"3"` still sends `"1"`; new RL looks for real ActionId `1` and rejects it (or could select a different action if ID `1` exists).
- this PR + pre-#10: selecting public `"3"` sends `"3"`; old RL treats it as index 3 and rejects it for a two-action Decision.
- this PR + RL #10: selecting public `"3"` sends `"3"`; new RL matches and executes ActionId `3`.

## Merge and deployment rule

Treat the RL #10 head and this Training PR head as one compatibility unit.

1. Approve and validate both PRs independently.
2. Merge Training PR #28 first if it is still pending, because this is a stacked cleanup of the workaround introduced there.
3. After #28 lands, retarget this PR from `agent/beam-search-whole-run-gate` to `main` and confirm its diff contains only the workaround removal, audit simplification, tests, and this document.
4. Merge RL #10 and this Training PR as the same release cutover.
5. Do not run live Whole Run traffic while only one side of the new pair is deployed. If deployment cannot be atomic, drain/stop Whole Run traffic, deploy both pinned SHAs, then resume traffic.
6. Run a sparse-ID smoke case after deployment (Reward `[0, 3]` or Shop `[0, 10]`) and verify the wire `commit_action.action_id` equals the selected public ID.

## Rollback rule

Rollback must also restore a compatible pair. Do not roll back only one side.

- Roll back both to the pre-#10 RL + #28 Training compatibility pair, or
- keep both on the contract-correct RL #10 + this Training change.

## Scope of this Training change

- remove `_commit_action_id()` ordinal translation from `CombatDecisionEngine`;
- send `DecisionOutcome.chosen_action_id` unchanged for Whole Run and Combat;
- remove SelectionAudit's Whole Run ordinal compatibility state;
- keep `selected_action_id` as an explicit audit label, now resolved by exact ID equality;
- update sparse-ID regression tests to require unchanged public IDs on the wire.
