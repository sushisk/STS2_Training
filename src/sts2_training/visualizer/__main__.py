from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from sts2_training.visualizer.core import EventStore, ReplayLogError, read_jsonl
from sts2_training.visualizer.live import LiveRunConfig, LiveRunController
from sts2_training.visualizer.server import VisualizerApp, make_server


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sts2_training.visualizer",
        description="Slay the Spire 2-styled visualizer for STS2_Training selection logs.",
    )
    parser.add_argument("--bind", default="127.0.0.1", help="visualizer HTTP bind address")
    parser.add_argument("--ui-port", type=int, default=7878, help="visualizer HTTP port")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    sub = parser.add_subparsers(dest="mode", required=True)

    live = sub.add_parser("live", help="start a Whole Run from the UI and watch its log live")
    live.add_argument("--host", default="127.0.0.1", help="STS2_RL TCP host")
    live.add_argument("--port", type=int, default=8765, help="STS2_RL TCP port")
    live.add_argument("--connect-timeout", type=float, default=5.0)
    live.add_argument("--character-id", default="IRONCLAD")
    live.add_argument("--ascension", type=int, default=0)
    live.add_argument("--seed", type=int, default=None)
    live.add_argument("--decision-timeout", type=float, default=30.0)
    live.add_argument("--max-decisions", type=_positive_int, default=None)
    live.add_argument("--search-mode", default=None)
    live.add_argument("--beam-depth", type=_positive_int, default=None)
    live.add_argument("--log", type=Path, default=None, help="JSONL output path")

    replay = sub.add_parser("replay", help="replay an existing JSONL selection log")
    replay.add_argument("log", type=Path)
    return parser


def _default_log_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("data/visualizer") / f"live-{stamp}.jsonl"


def build_app(args: argparse.Namespace) -> VisualizerApp:
    if args.mode == "replay":
        records = read_jsonl(args.log)
        store = EventStore(records)
        return VisualizerApp(mode="replay", store=store, replay_path=str(args.log))

    store = EventStore()
    config = LiveRunConfig(
        host=args.host,
        port=args.port,
        connect_timeout_s=args.connect_timeout,
        character_id=args.character_id,
        ascension=args.ascension,
        seed=args.seed,
        decision_timeout_s=args.decision_timeout,
        max_decisions=args.max_decisions,
        search_mode=args.search_mode,
        beam_max_depth=args.beam_depth,
    )
    live = LiveRunController(store=store, config=config, log_path=args.log or _default_log_path())
    return VisualizerApp(mode="live", store=store, live=live)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        app = build_app(args)
    except ReplayLogError as exc:
        parser.error(str(exc))

    server = make_server(app, bind=args.bind, port=args.ui_port)
    address, port = server.server_address[:2]
    browser_host = "127.0.0.1" if address in {"0.0.0.0", "::"} else address
    url = f"http://{browser_host}:{port}/"
    print(f"STS2 visualizer ({args.mode}) listening on {url}")
    if args.mode == "live":
        print("Open the UI and press START RUN to begin the configured Whole Run.")
    else:
        print(f"Loaded {len(app.store)} events from {args.log}")
    if not args.no_browser:
        threading.Timer(0.15, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
