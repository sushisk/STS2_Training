from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from sts2_training.visualizer.dto_contract import present_event
from sts2_training.visualizer.live import LiveRunController
from sts2_training.visualizer.page import INDEX_HTML
from sts2_training.visualizer.store import EventStore


class VisualizerApp:
    def __init__(
        self,
        *,
        mode: str,
        store: EventStore,
        live: LiveRunController | None = None,
        replay_path: str | None = None,
    ) -> None:
        if mode not in {"live", "replay"}:
            raise ValueError("mode must be 'live' or 'replay'")
        if mode == "live" and live is None:
            raise ValueError("live mode requires a LiveRunController")
        self.mode = mode
        self.store = store
        self.live = live
        self.replay_path = replay_path

    def status(self) -> dict[str, Any]:
        if self.mode == "live":
            assert self.live is not None
            return {"mode": "live", **self.live.status()}
        return {
            "mode": "replay",
            "state": "ready",
            "event_count": len(self.store),
            "log_path": self.replay_path,
            "error": None,
            "result": None,
        }

    def events_after(self, cursor: int) -> dict[str, Any]:
        if self.mode == "live" and self.live is not None:
            self.live.refresh()
        pairs = self.store.after(cursor)
        events = [present_event(index, record) for index, record in pairs]
        next_cursor = pairs[-1][0] if pairs else cursor
        return {"events": events, "next_cursor": next_cursor, "event_count": len(self.store)}

    def start_live(self) -> tuple[HTTPStatus, dict[str, Any]]:
        if self.mode != "live" or self.live is None:
            return HTTPStatus.METHOD_NOT_ALLOWED, {"error": "not available in replay mode"}
        if not self.live.start():
            return HTTPStatus.CONFLICT, {**self.live.status(), "error": "run is already running"}
        return HTTPStatus.ACCEPTED, {"ok": True, **self.live.status()}


def make_server(app: VisualizerApp, *, bind: str = "127.0.0.1", port: int = 7878) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "STS2Visualizer/1.0"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                self._send_text(INDEX_HTML, content_type="text/html; charset=utf-8")
                return
            if parsed.path == "/api/status":
                self._send_json(HTTPStatus.OK, app.status())
                return
            if parsed.path == "/api/events":
                query = parse_qs(parsed.query)
                try:
                    cursor = int(query.get("after", ["-1"])[0])
                except ValueError:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "after must be an integer"})
                    return
                self._send_json(HTTPStatus.OK, app.events_after(cursor))
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlsplit(self.path)
            if parsed.path == "/api/live/start":
                status, body = app.start_live()
                self._send_json(status, body)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_text(self, text: str, *, content_type: str) -> None:
            body = text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: HTTPStatus, value: Any) -> None:
            body = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((bind, port), Handler)
