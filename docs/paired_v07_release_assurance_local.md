# DTO v0.7 pair release assurance — STS2_Training local specifics

Repo-local supplement to [paired_v07_release_assurance.md](paired_v07_release_assurance.md), which is a canonical file synced byte-identical with STS2_RL's copy. **This file is NOT synced** — STS2_RL maintains its own version of this file describing its own CI.

The PR-required GitHub-hosted job in this repository is `training-hosted-contract` (workflow `paired-v07.yml`, historical identifier `paired-v07-exact-pair`). It validates Training-only client, protocol, retry/replay, correlation, capability, transport, and other Emulator-independent regressions.

`scripts/attest_paired_v07.ps1` and the real-Emulator paired test in this repository are advisory/manual validation only (see the shared doc's "Execution-security boundary" section).
