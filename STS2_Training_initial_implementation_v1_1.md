# STS2_Training 初期実装具体化設計 v1.1

- 文書状態: 実装開始前レビュー版
- 作成日: 2026-08-04
- 対象: `STS2_Training` と `STS2_RL/TrainingAPI` の接続
- 基準契約: RL–Training Communication API v0.5
- 参照実装:
  - `TrainingAPI/api_runtime.py`
  - `TrainingAPI/server.py`
  - `TrainingAPI/dto.py`
  - `TrainingAPI/validation.py`
  - `TrainingAPI/instance_combat.py`
  - `TrainingAPI/instance_whole_run.py`
  - `TrainingAPI/masking.py`
  - `TrainingAPI/mock_training_client.py`
- 上位文書: `STS2_Training_detailed_design_v1_0.md`

---

## 0. 結論

初期実装では、TrainingAPIにHTTP Serverを追加しない。

同一Windowsホスト上で、1つのTraining Processが1つのstatefulなRL Runtimeを所有する現在の条件では、既存の`RLApiServerProcess`をTransport Adapterとして利用する方が、次の点で優れる。

1. TCP port、Server起動待ち、Firewall、認証、HTTP Server dependencyが不要。
2. Training ProcessとCLRを確実に別OS Processへ分離できる。
3. `RLApiServer`の逐次実行とEmulatorのsingleton制約を自然に維持できる。
4. HTTP JSON encode/decodeを追加せず、同一ホストの往復遅延を最小化できる。
5. Runtimeの生成・終了をTrainingのlifecycleへ組み込みやすい。

ただし、Training本体を`RLApiServerProcess`へ直接依存させない。`RlTransport` Protocolを定義し、初期実装を`LocalProcessTransport`、将来実装を`HttpTransport`とする。

HTTP Serverを追加する条件は次に限定する。

- TrainingとRL Runtimeを別host／別containerへ配置する。
- RL RuntimeをTrainingとは独立して常駐・再起動・監視する必要が生じる。
- Python以外のClientからTrainingAPIを利用する。
- 複数Training Actorをnetwork越しに接続する設計へ移行する。

デバッグ目的だけではHTTPを導入しない。Request／Response Trace、Replay CLI、child process logを追加する方が、設計を増やさず同等以上の調査性を得られる。

---

## 1. 現行TrainingAPIから確定した事実

### 1.1 Process境界

`RLApiServerProcess`は`multiprocessing.get_context("spawn")`で子Processを作成し、子Process内だけで`RLApiServer`とCombat／Whole Run実装をimportする。

したがって、Training側が`RLApiServerProcess`を生成しても、Training Processではpythonnet／CLRを初期化しない。

Queueを通過する値はplainなJSON-safe `dict`であり、CLR objectやTrainingAPI内部classは越境しない。

### 1.2 Request処理

`RLApiServer`は次を行う。

1. `validate_request()`で共通・Operation別fieldを検証する。
2. `request_id`を`RequestLedger`で検査する。
3. `instance_id`から`CombatInstance`または`WholeRunInstance`へdispatchする。
4. 共通Response envelopeを付与する。

### 1.3 Combat

- rootは`LiveCombatSession`としてRL Runtime Process内に保持される。
- Branchは既存`BranchManager`／`BranchWorkerPool`へdispatchされる。
- `emulate_action`で有効な停止条件は`next_decision`だけである。
- Combat RNG Hypothesisは`rng_id`から安定したHypothesis indexへ写像される。
- 非root Branchをさらに分岐するときは、親と同じ`rng_id`が必要である。
- Branch処理は内部的には非同期になり得て、`emulate_action`が`running`を返す可能性がある。

### 1.4 Whole Run

- rootは`WholeRunSession`としてRL Runtime Process内に保持される。
- 現行コードは`enable_god_mode_for_testing()`を無条件で呼んでいる。
- `emulate_action`はActive Event boundaryだけで許可される。
- Map、Reward、Shop、Rest、Encounter、Boss／Ancient関連boundaryでは、正の`rng_id`を伴う`emulate_action`が`rng_hypothesis_unsupported_at_boundary`で拒否される。
- Whole Run Branch dispatchは同期実行である。
- Branchが新しい`map_select`へ到達した後、その非root Branchからさらに分岐することはできない。
- `stop_condition`は`next_decision`だけである。

### 1.5 Masked DTO

`masking.py`は、raw stateを再帰的にscrubする。

- forbidden key nameを再帰的に削除する。
- `drawPile`、`discardPile`、`exhaustPile`をMultisetへ変換する。
- `playPile`を削除する。
- `reward`を削除する。
- `Metrics`／`Extras`／`Info`は現在空allowlistである。
- `dto_version="emulator-fca2f06"`と`mask_version="1.0"`を付与する。

TrainingはこのDTOを完全schemaとして信用せず、Adapter側で使用fieldをallowlistする。

---

## 2. 初期実装前に修正すべきTrainingAPI

以下をPhase R0として、STS2_Training実装開始前または並行して修正する。

### 2.1 P0: Whole RunのGod Modeを無効化

対象: `TrainingAPI/instance_whole_run.py`

現行の次の処理を本番経路から除去する。

```python
self._session.enable_god_mode_for_testing()
```

必要なら`instance_config`へ本番では指定不能なtest専用flagを追加するのではなく、test factoryから明示注入する。

推奨形:

```python
class WholeRunInstance:
    def __init__(
        self,
        instance_id: str,
        instance_config: dict,
        *,
        session_factory: Callable[[], WholeRunSession] = WholeRunSession,
        enable_test_god_mode: bool = False,
        ...,
    ) -> None:
        self._session = session_factory()
        if enable_test_god_mode:
            self._session.enable_god_mode_for_testing()
```

Production bootstrapは必ず`enable_test_god_mode=False`とする。

### 2.2 P0: terminal結果を共通化

現状はCombat terminalが`terminal/outcome`、Whole Run terminalが`run_terminal`だけであり、Run勝敗Labelを確実に取得できない。

全Decision DTOへ次の共通fieldを追加する。

```json
{
  "boundary": "run_terminal",
  "terminal": true,
  "terminal_kind": "run",
  "outcome": "win",
  "legal_actions": []
}
```

規則:

- `terminal`: boolean、全Decision DTOに必須。
- `terminal_kind`: `null | "combat" | "run"`。
- `outcome`: `null | "win" | "loss"`。不明な値をTrainingへ推測させない。
- terminalでは`legal_actions=[]`。
- 非terminalでは`terminal=false`、`outcome=null`。

Trainingは完了したrootの`terminal_kind="run"`かつ`outcome in {win, loss}`だけを教師Labelとして採用する。

### 2.3 P0: BoundaryをCombat／Whole Runで共通化

Whole Runは`boundary`を公開しているが、Combatの`_decision_response_fields()`は明示追加していない。

Combatでも必ず次をextraへ入れる。

```python
extra={
    "boundary": view.boundary,
    "terminal": False,
    "terminal_kind": None,
    "outcome": None,
    "legal_actions": mask_legal_actions(...),
}
```

Trainingはraw state内の別fieldからBoundaryを推測しない。

### 2.4 P0: Public Action IDを一貫して発行

TrainingはAction内容を再構築せず、返された`action_id`をそのまま返す必要がある。

現行実装はDecision View側でindex文字列を解決する一方、公開DTOではraw Actionの`action_id`を残しており、両者が一致することを暗黙に仮定している。

新規classを追加する。

```python
@dataclass(frozen=True, slots=True)
class PublicActionEntry:
    public_action_id: str
    raw_index: int
    raw_action: dict

class PublicActionCatalog:
    def __init__(self, raw_actions: list[dict]) -> None: ...
    def public_actions(self) -> list[dict]: ...
    def resolve(self, public_action_id: str) -> tuple[int, dict]: ...
```

Decisionごとに次のようなIDを発行する。

```text
a-0000
a-0001
a-0002
```

このIDはDecision内だけで有効である。Card ID、room ID、raw action indexをpublic IDとして直接利用しない。

`Combat._DecisionView`と`WholeRun._View`は同じ`PublicActionCatalog`を所有する。

### 2.5 P0: Simulation CapabilityをResponseへ追加

TrainingがBoundary名や`fault_kind`からSimulation可能性を推測しないよう、Decision Responseへoptionalな`capabilities`を追加する。

```json
{
  "capabilities": {
    "can_emulate": true,
    "supported_stop_conditions": ["next_decision"],
    "rng_hypothesis": "combat_draw_order",
    "supports_async_result": true,
    "supports_branch_chaining": true
  }
}
```

Whole Run Active Event以外の例:

```json
{
  "capabilities": {
    "can_emulate": false,
    "supported_stop_conditions": [],
    "rng_hypothesis": "unsupported",
    "supports_async_result": false,
    "supports_branch_chaining": false
  }
}
```

v0.5 validationは未知top-level fieldを許容するため、Response拡張として導入できる。将来契約更新時に正式fieldへ昇格する。

### 2.6 P0: Combatの`running` Branchを回収可能にする

現行`CombatInstance.emulate_action()`は、pollで結果を取得できなければ`running`を返すが、後続`get_branch_status()`／`get_decision()`に結果をDecision Viewへmaterializeする処理がない。

次を追加する。

```python
class CombatInstance:
    def _harvest_completed(self, public_branch_ids: Iterable[str] | None = None) -> None:
        ...

    def _materialize_result(self, branch_id: str, result: BranchResult) -> None:
        ...
```

呼出箇所:

- `emulate_action()`のpoll後
- `get_branch_status()`の冒頭
- `get_decision()`の冒頭
- `cancel_branches()`／`release_branches()`前

これによりTrainingは`max_time_ms=0`で候補を短時間にsubmitし、Branch Workerを並列利用できる。

### 2.7 P0: Local Process Transportのtimeout後Responseを処理

現行`RLApiServerProcess.call()`は、timeout後に遅れて到着したResponseがQueueへ残る。次のcallがそのResponseを受け取ると`out-of-order response`になる。

`api_runtime.py`へ次を追加する。

- `threading.Lock`で`call()`をsingle callerに限定する。
- `internal_id`別のpending response map。
- deadlineまでQueueを読み、別IDならmapへ保存する。
- timeoutしたIDをexpired setへ登録し、遅延到着時は破棄またはtrace保存する。
- child Processが死亡した場合はdeadlineを待たずTransport例外にする。

推奨interface:

```python
class RLApiTransportError(RuntimeError): ...
class RLApiTransportTimeout(RLApiTransportError): ...
class RLApiRuntimeExited(RLApiTransportError): ...

class RLApiServerProcess:
    def call(self, payload: dict, *, timeout_s: float | None = None) -> dict:
        ...
```

Transport timeoutを`status="faulted"`の疑似Protocol Responseに変換しない。RL内部task timeoutとIPC timeoutを区別する。

### 2.8 P0: 1 Runtime ProcessにつきActive Instanceを1つに制限

Emulator側のstatic singletonとの衝突を避けるため、初期版では`RLApiServer`に同時に複数Instanceを保持させない。

`start_instance`時に`self._instances`が空でなければ`rejected`とする。

複数Actor化では、1 Actorにつき1 `RLApiServerProcess`を割り当てる。1 Runtime Process内へ複数rootを詰め込まない。

### 2.9 P1: `close_instance`再送を冪等化

現行はclose成功後にInstance Ledgerを削除するため、同じclose Requestを再送すると`unknown instance_id`になる。

次のどちらかを採用する。

- `closed_instance_responses[(instance_id, request_id)]`をRuntime生存中保持する。
- Instance Ledgerをtombstoneとして保持し、同じRequestをcache responseで返す。

初期推奨は後者である。

### 2.10 P1: Child Process Log

`_rl_runtime_process_main`へlog pathまたはlogging Queueを渡す。

最低限、次をJSONLで記録する。

- timestamp
- process pid
- request_id
- operation
- instance_id
- duration_ms
- status
- fault_kind
- exception type／traceback

DTO本体は既定では記録せず、digestとbyte sizeだけを記録する。Full payloadはdebug modeだけに限定する。

---

## 3. Transport Architecture Decision

### 3.1 Interface

```python
from typing import Any, Mapping, Protocol

JsonObject = dict[str, Any]

class RlTransport(Protocol):
    def call(self, request: Mapping[str, Any], *, timeout_s: float) -> JsonObject:
        ...

    def is_alive(self) -> bool:
        ...

    def close(self) -> None:
        ...
```

### 3.2 初期実装: `LocalProcessTransport`

```python
class LocalProcessTransport:
    def __init__(
        self,
        *,
        repo_root: Path,
        default_timeout_s: float,
    ) -> None: ...

    def call(self, request: Mapping[str, Any], *, timeout_s: float) -> JsonObject: ...
    def restart(self) -> None: ...
    def is_alive(self) -> bool: ...
    def close(self) -> None: ...
```

内部でのみ次をimportする。

```python
from TrainingAPI.api_runtime import RLApiServerProcess
```

TrainingのApplication／Decision／Model packageは`TrainingAPI`をimportしない。

### 3.3 Decorator: `TracingTransport`

```python
class TracingTransport:
    def __init__(self, inner: RlTransport, sink: TransportTraceSink) -> None: ...
    def call(self, request: Mapping[str, Any], *, timeout_s: float) -> JsonObject: ...
```

1 callごとに次を保存する。

```json
{
  "timestamp": "...",
  "request_id": "...",
  "operation": "emulate_action",
  "instance_id": "...",
  "branch_id": "...",
  "duration_ms": 12.4,
  "request_sha256": "...",
  "response_sha256": "...",
  "response_status": "completed",
  "fault_kind": null
}
```

`masked_emulator_dto`本体はEpisode fileへ保存し、Transport traceには重複保存しない。

### 3.4 将来実装: `HttpTransport`

初期版ではclass skeletonだけ用意してもよいが、dependencyは追加しない。

HTTP化時も`RLApiServer`を直接Web frameworkへ公開しない。1つのserialized command queueを挟み、同一Instanceへのhandler並列実行を禁止する。

推奨endpoint:

- `POST /v1/execute`
- `GET /health/live`
- `GET /health/ready`

HTTP Serverはsingle workerとし、`/v1/execute`を1本のcommand executorへ流す。FastAPI／Uvicornを採用するかはHTTP導入時に決定する。

---

## 4. STS2_Training File構成

```text
STS2_Training/
├─ pyproject.toml
├─ uv.lock
├─ configs/
│  ├─ bootstrap.toml
│  └─ value_guided.toml
├─ src/
│  └─ sts2_training/
│     ├─ __init__.py
│     ├─ main.py
│     ├─ bootstrap.py
│     ├─ config.py
│     ├─ protocol/
│     │  ├─ __init__.py
│     │  ├─ constants.py
│     │  ├─ request_models.py
│     │  ├─ response_models.py
│     │  └─ validation.py
│     ├─ rl/
│     │  ├─ __init__.py
│     │  ├─ transport.py
│     │  ├─ local_process_transport.py
│     │  ├─ tracing_transport.py
│     │  ├─ client.py
│     │  ├─ ids.py
│     │  └─ errors.py
│     ├─ episode/
│     │  ├─ __init__.py
│     │  ├─ controller.py
│     │  ├─ lifecycle.py
│     │  └─ records.py
│     ├─ decision/
│     │  ├─ __init__.py
│     │  ├─ candidate_selector.py
│     │  ├─ branch_evaluator.py
│     │  ├─ bootstrap_policy.py
│     │  ├─ value_policy.py
│     │  └─ direct_policy.py
│     ├─ state/
│     │  ├─ __init__.py
│     │  ├─ legal_action.py
│     │  ├─ decision_state.py
│     │  ├─ adapter.py
│     │  ├─ canonical.py
│     │  ├─ vocabulary.py
│     │  └─ encoder.py
│     ├─ model/
│     │  ├─ __init__.py
│     │  ├─ value_model.py
│     │  └─ predictor.py
│     ├─ data/
│     │  ├─ __init__.py
│     │  ├─ trajectory_writer.py
│     │  ├─ replay_catalog.py
│     │  └─ replay_buffer.py
│     ├─ train/
│     │  ├─ __init__.py
│     │  ├─ dataset.py
│     │  ├─ trainer.py
│     │  └─ metrics.py
│     ├─ checkpoint/
│     │  ├─ __init__.py
│     │  └─ manager.py
│     └─ observability/
│        ├─ __init__.py
│        ├─ logging.py
│        └─ trace.py
├─ tests/
│  ├─ unit/
│  ├─ contract/
│  ├─ integration/
│  └─ e2e/
└─ TrainingData/
   ├─ episodes/
   ├─ replay.sqlite3
   ├─ checkpoints/
   ├─ logs/
   └─ traces/
```

### 4.1 Dependency方向

```text
main/bootstrap
    ↓
episode/application
    ↓
decision ─── state ─── model
    ↓
rl client interface
    ↓
rl transport adapter
```

禁止:

- `model`から`TrainingAPI`をimportする。
- `episode`から`RLApiServerProcess`を直接生成する。
- `state`からHTTP／multiprocessingを参照する。
- `data`からEmulator objectを保存する。

---

## 5. 使用Library

### 5.1 Runtime

| Library | 用途 |
|---|---|
| Python 3.12 | Runtime |
| PyTorch | Value Model、Optimizer、inference |
| NumPy | RNG、数値処理 |
| Pydantic v2 | Configとwire boundary validation |
| 標準`json` | 初期JSONL保存とcanonical digest |
| 標準`sqlite3` | Replay catalog |

初期版では次を導入しない。

- `httpx`
- FastAPI／Uvicorn
- `orjson`
- Hydra
- Ray／Celery
- pandas／PyArrow
- pythonnet

JSON書込みが実測でbottleneckになった場合だけ`orjson`を追加する。

### 5.2 Development

- pytest
- pytest-cov
- Hypothesis
- mypy strict
- Ruff

---

## 6. Protocol Model

### 6.1 方針

RequestはTraining側で厳密に生成する。Responseは将来field追加を許容する。

Pydantic設定:

- Request: `extra="forbid"`
- Response envelope: `extra="allow"`
- `masked_emulator_dto`: `dict[str, Any]`
- internal domain object: frozen dataclass

### 6.2 Response Model

```python
class ApiResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: Literal["0.5"]
    request_id: str
    operation: Operation
    status: Status
    instance_id: str | None = None
    branch_id: str | None = None
    parent_branch_id: str | None = None
    rng_id: int | None = None
    decision_point_id: str | None = None
    branch_log: list[dict[str, Any]] | None = None
    masked_emulator_dto: dict[str, Any] | None = None
    branch_statuses: dict[str, Status] | None = None
    error: str | None = None
    fault_kind: str | None = None
```

Validation後、Operation別parserでDomain objectへ変換する。

### 6.3 Decision Domain Model

```python
@dataclass(frozen=True, slots=True)
class LegalAction:
    action_id: str
    action_type: str
    is_available: bool
    parameters: Mapping[str, Any]
    raw: Mapping[str, Any]

@dataclass(frozen=True, slots=True)
class SimulationCapabilities:
    can_emulate: bool
    supported_stop_conditions: tuple[str, ...]
    rng_hypothesis: str
    supports_async_result: bool
    supports_branch_chaining: bool

@dataclass(frozen=True, slots=True)
class DecisionState:
    instance_id: str
    branch_id: str
    decision_point_id: str
    boundary: str
    terminal: bool
    terminal_kind: str | None
    outcome: str | None
    legal_actions: tuple[LegalAction, ...]
    capabilities: SimulationCapabilities
    masked_emulator_dto: Mapping[str, Any]
    branch_log: tuple[Mapping[str, Any], ...]
```

Invariant:

- nonterminalなら`legal_actions`は1件以上。
- terminalなら`legal_actions`は空。
- `action_id`はDecision内で一意。
- `dto_version`と`mask_version`が対応versionである。

---

## 7. RL Client

### 7.1 Class

```python
class TrainingApiClient:
    def __init__(
        self,
        transport: RlTransport,
        request_ids: RequestIdGenerator,
        branch_ids: BranchIdGenerator,
        response_parser: ResponseParser,
        retry_policy: RetryPolicy,
    ) -> None: ...

    def start_instance(self, config: InstanceConfig) -> DecisionState: ...
    def get_decision(self, branch_id: str = "root") -> DecisionState | BranchPending: ...
    def commit_action(self, decision: DecisionState, action_id: str) -> DecisionState: ...
    def emulate_action(...) -> BranchSubmission: ...
    def get_branch_status(self, branch_ids: Sequence[str]) -> Mapping[str, Status]: ...
    def cancel_branches(self, branch_ids: Sequence[str]) -> None: ...
    def release_branches(self, branch_ids: Sequence[str]) -> None: ...
    def close_instance(self) -> None: ...
```

### 7.2 Client State

Clientが保持してよいstate:

- 現在の`instance_id`
- request counter／UUID generator
- branch ID generator
- transport health

Clientが保持しないstate:

- Snapshot
- Worker ID
- Lease
- RNG内部状態
- BranchのEmulator state

### 7.3 ID生成

初期版ではrandom UUIDではなく、Episode内で追跡しやすいIDを使う。

```text
request: req-e000123-000045
branch:  br-e000123-d000017-c003
```

同一Request再送では同じpayload objectと同じ`request_id`を使う。

### 7.4 Retry Rule

Operationを次の3群へ分ける。

#### Transport timeout時に同一Requestを1回再送

- `start_instance`
- `get_decision`
- `get_branch_status`
- `commit_action`
- `emulate_action`
- `cancel_branches`
- `release_branches`
- `close_instance`

ただし、これはPhase R0のTransport demultiplexとLedger冪等化が完了した後だけ有効にする。

2回目もTransport timeoutなら:

1. Runtime Processをterminateする。
2. Episodeを`aborted_transport_failure`として未完了保存する。
3. そのEpisodeをReplayへ入れない。
4. 新Runtime Processで次Episodeを開始する。

Protocol Responseの`status="faulted"`は自動再送しない。

---

## 8. Episode Controller

### 8.1 Class

```python
class EpisodeController:
    def __init__(
        self,
        api: TrainingApiClient,
        selector: CandidateSelector,
        branch_evaluator: BranchEvaluator,
        bootstrap_policy: BootstrapPolicy,
        value_policy: ValuePolicy,
        direct_policy: DirectPolicy,
        trajectory: TrajectoryWriter,
        rng: np.random.Generator,
    ) -> None: ...

    def run(self, spec: EpisodeSpec, model: ModelSnapshot | None) -> EpisodeSummary: ...
```

### 8.2 Main Loop

```text
start_instance
  ↓
parse DecisionState
  ↓ terminal?
  ├─ yes: finalize episode
  └─ no
      ↓
    choose policy path
      ├─ Bootstrap Mode
      ├─ Value-Guided + can_emulate
      └─ Direct Policy + cannot emulate
      ↓
    commit_action(root)
      ↓
    append root Decision record
      ↓
    repeat
```

### 8.3 Policy Path

#### Bootstrap Mode

- Branchを作らない。
- Legal Actionから決定論的seed付きrandomで選ぶ。
- root Trajectoryだけを収集する。

#### Value-Guided + `can_emulate=true`

- Candidateを選択する。
- Branchを作成する。
- terminal Branchは実Outcomeでscoreする。
- nonterminal BranchはValue Modelでbatch inferenceする。
- epsilon-greedyでActionを選ぶ。

#### Direct Policy + `can_emulate=false`

初期版ではLegal Actionからseed付きrandomで選ぶ。

これはWhole RunのMap／Reward／Shop／Rest等に適用される。Branch失敗として扱わず、`decision_mode="direct_unsupported_simulation"`を記録する。

後からBoundary別HeuristicまたはPolicy Modelを追加できるが、初期Value ModelのBranch評価と混同しない。

### 8.4 Commit後のBranch Cleanup

rootの`commit_action`成功時、RLは現行Decisionから派生した全BranchをCancel／Releaseする。

したがってTrainingは成功後に同じBranchへ`release_branches`を重ねて呼ばない。

次の場合だけ明示cleanupする。

- ActionをCommitする前にEpisodeを中断する。
- Candidate評価timeoutでfallbackする。
- Commitが`rejected`または`faulted`した。
- shutdown signalを受けた。

---

## 9. Candidate Selector

### 9.1 Interface

```python
class CandidateSelector(Protocol):
    def select(
        self,
        decision: DecisionState,
        *,
        limit: int,
        rng: np.random.Generator,
    ) -> tuple[LegalAction, ...]: ...
```

### 9.2 初期Algorithm

1. `is_available=false`を除外する。
2. Actionを`action_type`でgroup化する。
3. 各groupから最低1件をround-robinで選ぶ。
4. 残枠をgroup size比例で選ぶ。
5. group内順序はTraining RNGでshuffleする。
6. 同じDecision DTO、seed、limitなら同じ結果にする。

上限:

- Combat: 8
- Whole Run Active Event: 16
- その他: Branchを作らないためCandidate Selector不使用

Card／Relic tierや手作りscoreは初期Selectorへ入れない。

---

## 10. Branch Evaluator

### 10.1 Interface

```python
@dataclass(frozen=True, slots=True)
class BranchEvaluation:
    branch_id: str
    action_id: str
    status: str
    score: float | None
    terminal: bool
    outcome: str | None
    decision: DecisionState | None
    fault_kind: str | None
    error: str | None

class BranchEvaluator:
    def evaluate(
        self,
        root: DecisionState,
        candidates: Sequence[LegalAction],
        predictor: ValuePredictor,
    ) -> tuple[BranchEvaluation, ...]: ...
```

### 10.2 RNG

同じ親Decisionの全候補で`rng_id=1`を使用する。

これにより同じHypothesisでActionを比較する。

初期版では複数Hypothesis平均を行わない。

### 10.3 Combat

Phase R0のasync harvest修正後は次の順序にする。

1. 全候補を`max_time_ms=0`でsubmitする。
2. `running` Branch IDを集合へ入れる。
3. `get_branch_status(branch_ids)`をまとめてpollする。
4. completed Branchだけ`get_decision(branch_id)`で取得する。
5. overall deadline到達時に残BranchをCancel／Releaseする。
6. nonterminal DTOをまとめて1回のModel inferenceへ渡す。

Poll interval:

```text
10ms → 20ms → 50ms、以後50ms固定
```

### 10.4 Whole Run Active Event

現行WholeRunWorkerPoolは同期dispatchなので、候補を順に`emulate_action`する。

初期版でHTTPやThreadを追加して見かけ上並列化しない。Instance内部とregistryのthread safetyが保証されていないためである。

Whole Run Branch時間が問題になった場合、RL側へCombatと同じasync Branch Managerを追加する。

### 10.5 Branch Score

```text
terminal win  = 1.0
terminal loss = 0.0
nonterminal   = sigmoid(ValueModel(logit))
faulted       = scoreなし
rejected      = scoreなし
```

`partial`は初期版では受理しない。現行で`next_decision`以外を使わないためである。

### 10.6 全Branch失敗

- Candidate集合からseed付きrandomでActionを選ぶ。
- `fallback_reason="all_branches_failed"`を記録する。
- stale Decisionならfallbackせず、root Decisionを再取得してDecision全体をやり直す。

---

## 11. Decision Policy

### 11.1 Value Policy

```python
class ValuePolicy:
    def choose(
        self,
        evaluations: Sequence[BranchEvaluation],
        *,
        epsilon: float,
        rng: np.random.Generator,
    ) -> PolicyChoice: ...
```

規則:

1. scoreありの候補だけを対象にする。
2. probability `epsilon`で一様random。
3. それ以外は最大score。
4. 同点はAction IDで決めず、Training RNGで解決する。
5. 選択確率をTrajectoryへ保存する。

### 11.2 Direct Policy

Simulation不可boundary用。

初期版:

```python
class DirectPolicy:
    def choose(self, decision: DecisionState, rng: np.random.Generator) -> PolicyChoice:
        return uniform_random_available_action(...)
```

Value Modelの`V(s)`だけでは、BranchなしにAction間比較はできない。したがって、状態価値を各Actionへ誤って流用しない。

---

## 12. State Adapter

### 12.1 Version境界

```python
class StateAdapterRegistry:
    def resolve(
        self,
        *,
        dto_version: str,
        mask_version: str,
        instance_type: str,
    ) -> StateAdapter: ...
```

初期対応:

```text
(emulator-fca2f06, 1.0, combat)
(emulator-fca2f06, 1.0, whole_run)
```

未知versionはfail closedとする。

### 12.2 Allowlist

`masking.py`はHidden key除去を担うが、Training Encoderはさらに使用fieldを明示する。

Adapterは次を行う。

- 定義済みfieldだけをCanonicalStateへ写す。
- 未知fieldは無視し、raw DTOには保持する。
- 欠落optional fieldにはdefaultとpresence bitを付ける。
- Hidden情報を推測・再構成しない。
- list順序が意味不明なものはset／multisetとして扱う。

### 12.3 DTO Fixtureが必要

提供されたTrainingAPI codeはmask処理を定義しているが、Combat／Whole Runの完全なraw state field inventoryまでは定義していない。

Encoder実装前に、次のFixtureをRL Runtimeから取得してcommitする。

- Combat stable decision
- Combat pending target／selection decision
- Combat terminal win／loss
- Whole Run map_select
- reward
- shop
- rest
- event_choice
- combat entry／exit
- run terminal win／loss

各Fixtureには`dto_version`、`mask_version`、boundary、legal actionを含める。

---

## 13. Value Model初期版

上位設計を維持する。

```text
Numeric features
  └─ LayerNorm + Linear
Categorical entities
  └─ Embedding + EmbeddingBag(mean/sum)
Decision/Boundary
  └─ Embedding
Concatenate
  └─ MLP(256 → 128 → 1)
```

出力は1 logit。

```text
V(s) = P(final run win | masked public state)
```

Loss:

```python
torch.nn.BCEWithLogitsLoss()
```

初期版で行わないもの:

- TD target
- Branch pseudo label
- Policy head
- Reward shaping
- MCTS
- prioritized replay

---

## 14. Trajectory

### 14.1 保存対象

学習対象はroot上で実際にCommitされたDecisionだけである。

Branch stateは監査・評価logへ保存してよいが、教師datasetへ直接追加しない。

### 14.2 Decision Record

```json
{
  "record_type": "decision",
  "episode_id": "episode-000123",
  "step": 17,
  "instance_id": "inst-000001",
  "decision_point_id": "d-root-000018",
  "boundary": "combat_stable",
  "decision_mode": "value_guided",
  "masked_emulator_dto": {},
  "legal_actions": [],
  "candidate_action_ids": [],
  "branch_evaluations": [],
  "chosen_action_id": "a-0002",
  "selection_probability": 0.9,
  "epsilon": 0.1,
  "fallback_reason": null,
  "model_version": "model-000004",
  "training_seed_state_digest": "...",
  "final_outcome": null
}
```

Episode完了時にsummaryへOutcomeを保存し、Replay loaderが各Decisionへ同じLabelを付ける。大きなJSONL全行を書き換えない。

### 14.3 Incomplete Episode

書込み先:

```text
episode_000123.jsonl.inprogress
```

正常terminalと`close_instance`完了後だけ:

```text
episode_000123.jsonl
episode_000123.summary.json
```

へatomic renameする。

Transport crash、Training crash、未知terminal outcomeはReplay対象外とする。

---

## 15. Config

```toml
[rl]
transport = "local_process"
repo_root = "../STS2_RL"
request_timeout_s = 60.0
transport_retry_count = 1
runtime_restart_after_timeout = true

[episode]
instance_type = "whole_run"
character_id = "IRONCLAD"
ascension = 10
seed_start = 1

[decision]
combat_candidate_limit = 8
event_candidate_limit = 16
rng_id = 1
epsilon = 0.10
branch_deadline_s = 60.0
branch_poll_initial_ms = 10
branch_poll_max_ms = 50

[training]
bootstrap_episodes = 100
train_every_episodes = 10
batch_size = 256
checkpoint_every_episodes = 100

[model]
device = "auto"
embedding_dim = 32
hidden_dims = [256, 128]
learning_rate = 0.0003
weight_decay = 0.00001

[data]
root = "TrainingData"
validation_seed_modulus = 10
validation_seed_remainder = 0

[debug]
trace_transport = true
trace_full_payload = false
```

---

## 16. Bootstrap構築

```python
def build_application(config: TrainingConfig) -> TrainingApplication:
    base_transport = LocalProcessTransport(
        repo_root=config.rl.repo_root,
        default_timeout_s=config.rl.request_timeout_s,
    )
    transport: RlTransport = base_transport
    if config.debug.trace_transport:
        transport = TracingTransport(transport, JsonlTransportTraceSink(...))

    api = TrainingApiClient(
        transport=transport,
        request_ids=RequestIdGenerator(...),
        branch_ids=BranchIdGenerator(...),
        response_parser=ResponseParser(...),
        retry_policy=RetryPolicy(...),
    )

    adapter_registry = build_adapter_registry()
    vocabulary = VocabularyStore.load_or_create(...)
    encoder = MaskedDtoEncoder(adapter_registry, vocabulary, ...)
    model = ValueModel(...)
    predictor = ValuePredictor(model, encoder, ...)

    return TrainingApplication(
        episode_controller=EpisodeController(...),
        trainer=Trainer(...),
        checkpoint_manager=CheckpointManager(...),
    )
```

DI frameworkは使用しない。

---

## 17. Exception設計

```text
TrainingError
├─ ConfigurationError
├─ ProtocolError
│  ├─ ResponseValidationError
│  ├─ SchemaVersionMismatch
│  ├─ MaskVersionMismatch
│  └─ ProtocolInvariantError
├─ TransportError
│  ├─ TransportTimeout
│  └─ RuntimeExited
├─ ApiOperationError
│  ├─ RequestRejectedError
│  └─ RequestFaultedError
├─ EpisodeAborted
└─ StorageError
```

`rejected`と`faulted`を同じ例外にしない。

- `rejected`: Request、stale ID、capability、validation問題。
- `faulted`: 実行開始後のEmulator／Worker問題。
- Transport exception: Protocol Responseを受け取れていない。

---

## 18. Test計画

### 18.1 TrainingAPI修正のContract Test

1. Whole Run production instanceでGod Modeが有効にならない。
2. Combat／Whole Runの全Decisionに`boundary`がある。
3. root run terminalに`outcome=win|loss`がある。
4. terminalの`legal_actions=[]`。
5. 公開Action IDをそのままCommit／Emulateできる。
6. Action IDはDecision内で一意。
7. `capabilities.can_emulate`と実際の受理可否が一致する。
8. Combat Branchを`running`から`completed`へpoll後、`get_decision`できる。
9. Transport timeout後の遅延Responseが次Requestを壊さない。
10. 2つ目のActive Instanceが拒否される。
11. 同じ`close_instance` Request再送が同じResponseを返す。

### 18.2 Training Unit Test

- Request model serialization
- Response invariant validation
- IDの非再利用
- Candidate group samplingの決定性
- epsilon-greedyの確率
- unsupported simulationでDirect Policyへ移る
- terminal Branch score
- all-branch-failed fallback
- unknown DTO／mask version fail closed
- episode-uniform Replay sampling
- incomplete episode exclusion

### 18.3 Integration Test

実物`RLApiServerProcess`を使用する。

- Training Processで`clr`／`pythonnet`がimportされない。
- Combat start → emulate複数 → poll → batch score stub → commit → close。
- Whole Run start → unsupported boundaryではDirect Policy → Eventではemulate → commit。
- stale Decisionを再利用するとrejected。
- Commit成功後Branchがreleasedになる。
- Runtime killでEpisodeがabortedになり、次Episodeで再生成できる。

### 18.4 E2E

#### E2E-Bootstrap

- 1 Whole RunをModelなしで終了まで進行。
- 完了Episode JSONL／summaryを作成。
- Outcome LabelをReplayへ登録。

#### E2E-Train

- 完了EpisodeからValue Modelを1 updateする。
- Checkpointを保存・再読込する。

#### E2E-Value-Guided

- Combat Decisionで2件以上のBranchを作成する。
- nonterminal Branchを1 batchで推論する。
- 最大score ActionをrootへCommitする。

---

## 19. 実装順序

### Phase R0: TrainingAPI Hardening

成果物:

- God Mode除去
- terminal／boundary共通化
- PublicActionCatalog
- capabilities
- Combat async harvest
- Transport timeout demultiplex
- single active instance
- close tombstone
- Contract tests

停止条件:

- P0 Contract Testが1件でも失敗する場合、Value-Guided Trainingへ進まない。

### Phase T0: Skeleton／Local Transport

成果物:

- `pyproject.toml`
- Config
- `RlTransport`
- `LocalProcessTransport`
- `TracingTransport`
- `TrainingApiClient`
- Protocol models／parser

受け入れ:

- Training parent ProcessでCLR未import。
- Combat／Whole Runをstart／get／closeできる。

### Phase T1: Bootstrap Episode

成果物:

- `EpisodeController`
- `BootstrapPolicy`
- `DirectPolicy`
- `TrajectoryWriter`
- 完了／未完了Episode lifecycle

受け入れ:

- Branchを一切作らずWhole Runを完了できる。
- 正しいwin／loss Labelを保存できる。

### Phase T2: DTO Fixture／Encoder／Value Model

成果物:

- Boundary別DTO Fixture
- State Adapter v1
- Vocabulary
- Encoder
- Value Model
- Replay／Trainer／Checkpoint

受け入れ:

- 100 Bootstrap Episodeから学習できる。
- Checkpoint再開結果が一致する。

### Phase T3: Combat Value-Guided

成果物:

- CandidateSelector
- BranchEvaluator
- ValuePolicy
- Combat async submit／poll
- batch inference

受け入れ:

- 複数Branch Workerを同時利用する。
- rootがBranch評価中に変化しない。
- Branch stateを教師Labelに使わない。

### Phase T4: Whole Run Active Event Value-Guided

成果物:

- Event boundaryだけBranch評価
- unsupported boundaryのDirect Policy
- capability mismatch監査

受け入れ:

- unsupported boundaryで不要な`emulate_action`を送らない。
- Event候補を同一`rng_id=1`で比較する。

### Phase T5: Hardening

成果物:

- Transport crash recovery
- Metric／Calibration
- Checkpoint retention
- Replay integrity audit
- 長時間E2E

---

## 20. HTTP移行判断Gate

次のいずれかが確定するまでHTTP Serverを実装しない。

1. TrainingとRLを別hostへ配置する設計が承認された。
2. RL Runtimeを独立serviceとして運用する必要がある。
3. 複数Actor向けのnetwork protocolが必要になった。

HTTP導入時の受け入れ条件:

- LocalProcessTransportと同じContract Test suiteを通る。
- 同一InstanceのRequestはserver側で直列化される。
- client disconnect後もrequest idempotencyが壊れない。
- localhost round-trip overheadを実測し、Decision全体の5%未満である。
- health endpointがCLR初期化完了とWorker readinessを区別する。
- local modeを削除せず、E2Eと開発用途に残す。

---

## 21. 最終判断

初期の最善設計は、HTTP Server追加ではなく、次の組合せである。

```text
STS2_Training
  → TrainingApiClient
  → TracingTransport
  → LocalProcessTransport
  → RLApiServerProcess (spawn)
  → RLApiServer
  → CombatInstance / WholeRunInstance
  → Branch Workers / Emulator
```

これにより、同一hostでの低遅延とProcess分離を維持しながら、Training側のTransport依存を1 Adapterへ閉じ込める。

優先度はHTTP化よりも、terminal Label、Action ID、God Mode、capability、Combat Branch回収、timeout後Response処理の修正が高い。これらを直さずTransportだけHTTPへ変更しても、学習の正しさと障害時の安全性は改善しない。
