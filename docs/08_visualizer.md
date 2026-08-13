# visualizer モジュール

## 0. 文章の目的

この文書は `src/sts2_training/visualizer/` の live/replay dashboard 実装を説明する。対象は JSONL log reader、in-memory store、presentation DTO、HTTP server、embedded browser asset、live runner controller である。

## 1. 概要

visualizer は self-play / runner が出力する JSONL run event log を browser で見るための軽量 web dashboard である。completed log を replay する mode と、runner process を起動して growing JSONL を tail する live mode を持つ。

browser-facing contract は `presentation.present_event()` と `dto_contract.present_event()` が作る dict で、raw DTO の current fields を優先しつつ、legacy alias も表示用に整形する。large embedded HTML/CSS/JS は `assets.py` に保持され、`page.py` が必要な focused replacement を適用して `INDEX_HTML` を作る。

## 2. Architecture

| ファイル | 役割 |
|---|---|
| `assets.py` | embedded `INDEX_HTML` asset |
| `page.py` | HTML asset の build hook |
| `presentation.py` | JSONL record を stable browser-facing event view に変換 |
| `dto_contract.py` | formal current Emulator DTO fields を優先する presenter |
| `log_reader.py` | growing JSONL を incremental に読む `JsonlLogReader` |
| `store.py` | thread-safe in-memory event store |
| `server.py` | `VisualizerApp` と `ThreadingHTTPServer` factory |
| `live.py` | runner process を起動し、log を tail する `LiveRunController` |
| `core.py` | split 後の backward-compatible imports |
| `__main__.py` | `python -m sts2_training.visualizer` CLI |

`JsonlLogReader.poll(final=False)` は complete line だけ decode し、growing file の途中行を待つ。decode 不能な JSONL は `ReplayLogError`。`EventStore` は records を thread-safe に保持し、server endpoint から snapshot と incremental update を返せる。

`LiveRunController` は既存 Whole Run CLI を subprocess として起動し、`--log` が未指定なら default log path を補う。`status()` は process/log/event count を返す。

## 3. API

```python
class JsonlLogReader:
    def __init__(self, path: str | Path) -> None
    def poll(self, *, final: bool = False) -> list[dict[str, Any]]

read_jsonl(path: str | Path) -> list[dict[str, Any]]
```

```python
class EventStore:
    def append(self, record: Mapping[str, Any]) -> None
    def clear(self) -> None
    def after(self, cursor: int) -> list[tuple[int, dict[str, Any]]]
```

```python
class VisualizerApp:
    def __init__(self, *, mode: str, store: EventStore, live: LiveRunController | None = None, replay_path: str | None = None)
    def status(self) -> dict[str, Any]
    def events_after(self, cursor: int) -> dict[str, Any]
    def start_live(self) -> tuple[HTTPStatus, dict[str, Any]]

make_server(app: VisualizerApp, *, bind: str = "127.0.0.1", port: int = 7878) -> ThreadingHTTPServer
```

```python
presentation.present_event(index: int, record: Mapping[str, Any]) -> dict[str, Any]
dto_contract.present_event(index: int, record: Mapping[str, Any]) -> dict[str, Any]
build_index_html() -> str
```

## 4. 使用例

completed log replay:

```bash
python -m sts2_training.visualizer replay data/runs/run-001.jsonl \
  --bind 127.0.0.1 \
  --ui-port 7878
```

live mode:

```bash
python -m sts2_training.visualizer live \
  --ui-port 7878 \
  --log data/visualizer/live.jsonl \
  -- --host 127.0.0.1 --port 8765 --character-id IRONCLAD --seed 123 --search-mode standard
```

Python から log を読む:

```python
from sts2_training.visualizer import read_jsonl
from sts2_training.visualizer.presentation import present_event

records = read_jsonl("data/runs/run-001.jsonl")
first_event = present_event(0, records[0])
```

## 5. 補足説明

visualizer は学習ロジックを持たず、既存 JSONL を表示用 DTO に変換するだけである。Oracle JSONL の意味は [04_oracle.md](04_oracle.md)、self-play や runner log の生成は [07_runner_cli.md](07_runner_cli.md) を参照する。
