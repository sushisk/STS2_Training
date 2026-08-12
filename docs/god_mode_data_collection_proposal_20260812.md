# Proposal: God Mode data-collection support in `self_play.py` (2026-08-12)

Status: **proposal only - no code changes in this PR.** Companion to
[STS2_RL#39](https://github.com/sushisk/STS2_RL/pull/39), which proposes the RL-side
opt-in and the Emulator-side DTO exposure this depends on. Raised for review before
implementation starts.

## 1. Motivation

Full combat-logic implementation is underway and needs start-of-combat state diversity
(decks, relics, enemy encounters) across floors/Acts. Current no-beam self-play averages
floor ~10 (recent 50/100-run IRONCLAD batches: `floor_stats.mean` 10.04 / 10.08) and
rarely reaches Act 2+, so natural play doesn't sample deep-run states often enough.

The proposed fix is collecting a separate batch of Whole Runs with the player invincible
(Emulator's `EnableGodModeForTesting()`, applied via the RL-side opt-in proposed in
STS2_RL#39) so runs reach deep floors and build out large, varied decks quickly. Reward
generation itself is unaffected by God Mode (confirmed by reading the Emulator
implementation this session - it only applies Strength/Buffer/Regen powers to the
player's Creature, never touching reward selection/RNG). What God Mode *does* corrupt -
`hp` pinned near `maxHp`, `playerPowers` showing the applied god-mode stacks, and the
in-combat play-by-play itself - must never reach training as-is.

## 2. Proposed design

**Principle, matching STS2_RL#39:** default OFF; explicit opt-in; two independent signals
(directory placement + an in-record flag) mark god-mode data, so a filtering mistake in
one doesn't silently leak contaminated records into a normal training set.

### `self_play.py`
- New `--god-mode` CLI flag. When set:
  - Passes `god_mode=True` through to `start_instance`'s instance config (the RL-side
    surface proposed in STS2_RL#39; this flag is inert until that lands).
  - Defaults `--output-dir` to a distinct subdirectory, e.g. `data/self_play/godmode/`,
    instead of silently reusing the normal collection path. An explicit `--output-dir`
    still overrides this, but the *default* must not be the shared path.
- No change to the normal (non-god-mode) collection path or its default output directory.

### Log record flag
- Once the Emulator exposes `godMode` in `masked_emulator_dto` (STS2_RL#39 / Emulator
  side), every JSONL record already carries `received.masked_emulator_dto.godMode`
  automatically - `JsonlSelectionLogger` needs no format change for this, since it logs
  the DTO verbatim. The only actionable change here is documenting this field in
  `docs/` (e.g. this doc / the visualizer's `how_to_use.md`) so downstream readers know
  to check it rather than relying on directory placement alone.
- The final `self_play_run_result` record (from `run_result_event`) should also carry the
  flag explicitly at the top level (not just buried in `final_dto`), since that's the
  record most tooling (`floor_reach_eval`-style aggregation, the visualizer) reads first
  for run-level metadata.

### Correction / post-processing tool
- New standalone tool (e.g. `tools/correct_god_mode_logs.py`, mirroring
  `tools/score_cards.py`'s existing standalone-tool pattern) that:
  - Reads JSONL from a god-mode output directory.
  - Refuses to process a file whose records don't carry `godMode: true` (fails loudly
    rather than silently correcting - and therefore silently *validating* - a normal run).
  - Strips the specific applied powers (Strength/Buffer/Regen) from every
    `playerPowers` occurrence in each record.
  - Writes to a distinct output directory (e.g. `data/self_play/godmode_corrected/`),
    never mutating the raw collection in place.
- **Explicitly out of scope for this pass:** the replacement policy for `hp`/`maxHp`
  (uniform random? floor-conditioned? something else?) is a separate, still-open design
  question - the tool above does not attempt it. `playerPowers` stripping alone is the
  well-defined part.

## 3. Dependencies / sequencing

This is inert without STS2_RL#39's opt-in and the Emulator's `godMode` DTO field - none of
`self_play.py`'s `--god-mode` flag does anything server-side until those land. The
correction tool can be written independently since it only operates on already-collected
JSONL, but has nothing real to correct until collection is possible.

## 4. Open questions for review

1. Should `--god-mode` be usable together with `--search-mode`/beam search at all, or
   should it force `--no-beam` (beam search's value function wasn't designed with an
   invincible player in mind, and beam evaluation cost buys nothing when death isn't a
   real outcome to search around)?
2. Does the correction tool belong in `tools/` (standalone, matching `score_cards.py`) or
   as a `sts2_training.runner`/`sts2_training.selection` module with an accompanying test
   suite, given it will presumably run repeatedly as new god-mode batches are collected?
3. HP/`maxHp` correction policy (see §2, "explicitly out of scope") needs its own design
   pass before corrected data is usable for training - follow-up PR gated on this one, or
   designed in parallel?

No code changes are included in this PR. Filed for review of the approach before
implementation begins.
