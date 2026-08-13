# DTO v0.7 paired release gate

DTO v0.7 is a hard-cutover wire-contract change shared by STS2_Training and STS2_RL. The two repositories therefore need two different kinds of assurance, and they must not be conflated.

## Repo-local required CI

Each repository has its own PR-required GitHub-hosted job that validates only that repository's own Emulator-independent regressions. A green repo-local check does **not** prove that a particular counterpart-repository commit is compatible with this commit.

Each repository's workflow file keeps a historical `paired-v07-*` identifier only for tooling compatibility. That identifier must not be interpreted as cross-repository attestation.

The exact required job name, workflow file, and what it validates are repo-local facts, not part of this shared contract - see this repository's own `dto_paired_release_gate_v0.7_local.md` (not synced across repositories).

## Exact-pair release gate

A trusted paired gate, when used for release/deployment, must identify the exact pair `(training_sha, rl_sha)` and satisfy all of the following:

1. Resolve both PR/deployment SHAs before test execution.
2. Check out both repositories by immutable SHA, never by moving branch or PR refs.
3. Run paired wire/integration validation against that exact pair.
4. Re-read both source heads after validation and discard the result if either head moved.
5. Bind the result to both SHAs so a green result cannot be reused after one side changes.
6. Re-evaluate whenever either counterpart head changes.
7. Deploy the attested pair as one pinned compatibility unit.

If independent rolling deployment is required, v0.7 hard cutover is insufficient; dual-version support or explicit version/capability negotiation is required first.

## Execution-security boundary

Real-Emulator paired tests execute PR-controlled code and therefore require a stronger boundary than removing GitHub credentials from the test process. A trusted release gate should run that code in a disposable/ephemeral or equivalently isolated worker with no GitHub write credential, no developer credential store, minimal filesystem exposure, and restricted network access. A separate trusted controller may hold status-publishing credentials and publish a result only after verifying the pair identity and returned test evidence.

Until such an isolated exact-pair orchestrator exists, any real-Emulator paired-validation script in either repository is advisory/manual validation only and must not be represented as a branch-protection proof of exact-pair compatibility. See this repository's own `dto_paired_release_gate_v0.7_local.md` for the exact script/test this repository runs.

## Worked example: Whole Run `action_id` cutover (Training #28 / `sushisk/STS2_RL#10`)

A completed instance of the exact-pair release gate above, kept as precedent for future paired hard-cutovers.

Training PR #28 contained a temporary compatibility workaround for the pre-RL-#10 Whole Run server: selection still used the public sparse `action_id`, but root `commit_action` translated that ID to its ordinal position before sending it on the wire. RL #10 fixed the server contract instead: Whole Run resolves the wire token by exact `action_id` equality against the current legal actions, matching the DTO rule that Training returns the published ID unchanged. The two fixes were not independently deployable:

| Training | RL | Result |
| --- | --- | --- |
| #28 compatibility behavior | pre-#10 | Supported temporary pair |
| #28 compatibility behavior | #10 | **Unsupported**: ordinal may be rejected or select a different real ActionId |
| post-#28 cutover | pre-#10 | **Unsupported**: sparse public ID is still interpreted as a positional index |
| post-#28 cutover | #10 | Supported contract-correct pair |

For example, a Reward Decision with public IDs `["0", "3"]` illustrates both failure directions:

- #28 + pre-#10: selecting public `"3"` sends ordinal `"1"`; old RL indexes slot 1 and executes ActionId `3`.
- #28 + RL #10: selecting public `"3"` still sends `"1"`; new RL looks for real ActionId `1` and rejects it (or could select a different action if ID `1` exists).
- post-#28 cutover + pre-#10: selecting public `"3"` sends `"3"`; old RL treats it as index 3 and rejects it for a two-action Decision.
- post-#28 cutover + RL #10: selecting public `"3"` sends `"3"`; new RL matches and executes ActionId `3`.

Deployment followed the exact-pair release gate: both PRs were approved/validated independently; Training #28 (the temporary workaround) landed first, since the cutover PR was a stacked cleanup of the workaround #28 introduced; once #28 landed, the cutover PR was retargeted to `main` and its diff was confirmed to contain only the workaround removal, audit simplification, tests, and this document; then RL #10 and the Training cutover PR were merged as one release unit. No live Whole Run traffic ran while only one side of the pair was deployed. A sparse-ID smoke case (Reward `[0, 3]` or Shop `[0, 10]`) verified the wire `commit_action.action_id` equaled the selected public ID after deployment. The corresponding rollback rule: roll back both sides to the pre-#10 RL + #28 Training compatibility pair, or keep both on the contract-correct pair - never roll back only one side.
