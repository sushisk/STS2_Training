# Log visualizer

The visualizer turns `selection_log.JsonlSelectionLogger` records into a local, browser-based
combat view. It intentionally ships with no frontend/runtime dependency: the Python process
serves a self-contained HTML/CSS/JS page from the standard library HTTP server.

The visual language follows Slay the Spire 2's combat layout (dark dungeon field, compact top
resource bar, player/enemy health bars, bottom fanned hand, energy gem, highlighted selected
card/action) without bundling or copying game assets. Training-only metadata is kept in side
panels so it does not obscure the board.

Both modes use the same input boundary:

```text
Runner -> JSONL selection log -> Visualizer reader -> present_event -> Browser
```

`present_event` is the only DTO-to-browser adapter. Replay reads a completed JSONL file; live
mode tails a growing one and keeps an unterminated final line buffered until the next poll.

## Live mode

Live mode does not construct an API client or duplicate Whole Run configuration. Pressing
**START RUN** launches the existing `sts2_training.runner.start_new_run` CLI as a subprocess,
with its arguments passed through unchanged and `--selection-log` supplied by the visualizer.

Start STS2_RL first, then start the visualizer. Put runner arguments after `--`:

```bash
python -m sts2_training.visualizer live \
  --log data/visualizer/ironclad.jsonl \
  -- \
  --host 127.0.0.1 --port 8765 \
  --character-id IRONCLAD --ascension 0
```

Open the printed URL and press **START RUN**. The normal runner owns `TcpConnection`,
`AsyncTrainingApiClient`, all run/search defaults, and JSONL logging. The visualizer only
starts that entry point and tails the file, so changes to runner defaults do not require a
second set of visualizer options.

`--log` is optional; live mode otherwise creates a timestamped file under `data/visualizer/`.
Do not pass `--selection-log` after `--`; that path is managed by the visualizer.

The runner can also produce a visualizable log without the visualizer:

```bash
python -m sts2_training.runner.start_new_run \
  --host 127.0.0.1 --port 8765 \
  --character-id IRONCLAD \
  --selection-log data/runs/ironclad.jsonl
```

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
- `GET /api/status` - mode, lifecycle, event count, log path, live runner exit/error
- `GET /api/events?after=N` - normalized events after zero-based cursor `N`
- `POST /api/live/start` - launch the configured runner CLI (live mode only, once per process)

The HTTP server binds to `127.0.0.1:7878` by default. Use `--bind` / `--ui-port` to change it
and `--no-browser` for headless environments.
