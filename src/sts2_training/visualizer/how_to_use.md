# Log visualizer

The visualizer turns `selection_log.JsonlSelectionLogger` records into a local, browser-based
combat view. It intentionally ships with no frontend/runtime dependency: the Python process
serves a self-contained HTML/CSS/JS page from the standard library HTTP server.

The visual language follows Slay the Spire 2's combat layout (dark dungeon field, compact top
resource bar, player/enemy health bars, bottom fanned hand, energy gem, highlighted selected
card/action) without bundling or copying game assets. Training-only metadata is kept in side
panels so it does not obscure the board.

## Live mode

Start STS2_RL first, then start the visualizer:

```bash
python -m sts2_training.visualizer live \
  --host 127.0.0.1 --port 8765 \
  --character-id IRONCLAD --ascension 0
```

Open the printed URL and press **START RUN**. The visualizer creates the same
`AsyncTrainingApiClient` + `start_new_run` pipeline as the normal runner. Its selection logger
is tee'd to a timestamped JSONL file under `data/visualizer/` and to the browser event stream,
so committed actions and beam-search branches appear while the run is progressing.

Use `--log path/to/run.jsonl` to choose the live output file, and pass the same `--seed`,
`--search-mode`, `--beam-depth`, `--decision-timeout`, and `--max-decisions` controls when a
reproducible run is needed.

## Replay mode

Replay any existing selection JSONL without STS2_RL running:

```bash
python -m sts2_training.visualizer replay data/self_play/<run>.jsonl
```

The transport bar supports play/pause, previous/next event, 0.5x-4x speed, and arbitrary
seeking. Clicking a timeline entry jumps directly to that event. `commit_action` selections
are visually separated from speculative `emulate_actions` branches.

The visualizer prefers `received.masked_emulator_dto` as the state on which the action was
selected and keeps `result.masked_emulator_dto` in the event payload for inspection. A
`self_play_run_result.final_dto` record is also accepted as a terminal replay frame.

## Browser/API endpoints

- `GET /` - visualizer UI
- `GET /api/status` - mode, lifecycle, event count, log path, live-run result/error
- `GET /api/events?after=N` - normalized events after zero-based cursor `N`
- `POST /api/live/start` - start the configured Whole Run (live mode only, once per process)

The HTTP server binds to `127.0.0.1:7878` by default. Use `--bind` / `--ui-port` to change it
and `--no-browser` for headless environments.
