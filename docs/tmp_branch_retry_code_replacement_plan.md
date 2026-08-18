# Branch-local Retry 実装計画

## 目的

現在の `STS2_Training` では、`AllBranchesFaultedError` が発生した場合に `OracleEpisodeRunner` 側で探索全体を最大3回やり直しています。

今回の変更では、この挙動を以下へ置き換えます。

- 探索全体は1回だけ実行する
- 同一 parent / decision / action / RNG hypothesis に対応する candidate branch が `faulted` になった場合、その candidate だけを最大3回試行する
- retry ごとに fresh `branch_id` を発行する
- `completed` / `partial` になった candidate はそれ以上 retry しない
- 最大試行回数後も `faulted` の candidate のみを最終 fault として扱う
- 全 candidate が最大試行回数まで失敗した場合にだけ `AllBranchesFaultedError` を上位へ送る
- `OracleEpisodeRunner` 側の whole-search retry は削除する

---

## 変更対象ファイル

最初の実装では以下の4ファイルを変更対象とします。

1. `src/sts2_training/decision/beam_search.py`
2. `src/sts2_training/runner/oracle_collection.py`
3. `tests/decision/test_beam_search.py`
4. `tests/runner/test_oracle_collection.py`

wire protocol 自体は変更しません。既存の `emulate_actions` を使用し、retry は fresh `branch_id` を持つ新しい branch execution として送信します。

---

## 1. `BeamSearchConfig` に `max_branch_attempts` を追加

`BeamSearchConfig` に以下を追加します。

```python
@dataclass
class BeamSearchConfig:
    beam_width: int = 8
    top_k_actions: int = 4
    max_depth: int = 2
    ...
    max_branch_attempts: int = 3
```

`max_branch_attempts` は初回実行を含む総試行回数とします。

つまり、

```text
max_branch_attempts = 3
```

の場合は、

```text
attempt 1
attempt 2
attempt 3
```

の最大3回です。

既存の config validation にも `max_branch_attempts` を追加します。

```python
for name in (
    "beam_width",
    "top_k_actions",
    "max_depth",
    "max_batch_size",
    "max_continuation_steps",
    "max_branch_attempts",
):
    ...
```

Oracle 側は runtime config を `replace()` して config を生成しているため、基本的には追加の引き回しなしで同じ値を利用できます。

---

## 2. branch result 判定 helper を追加

`beam_search.py` に branch result が探索上の有効結果かを判定する helper を追加します。

```python
def _is_resolved_branch_result(result: Any) -> bool:
    return (
        isinstance(result, Mapping)
        and result.get("status") in _RESOLVED_STATUSES
    )
```

既存の `_RESOLVED_STATUSES` は以下を想定します。

```python
_RESOLVED_STATUSES = {"completed", "partial"}
```

したがって扱いは以下です。

```text
completed -> success
partial   -> success
faulted   -> retry candidate
```

batch-level の `RequestRejectedError` は branch retry の対象にしません。

---

## 3. `_emulate_depth_batch()` を attempt round 方式に変更

現在の `_emulate_depth_batch()` は candidate を1回だけ `emulate_actions()` に送り、結果をそのまま `_score_frontier()` へ渡しています。

これを以下の構造へ変更します。

```python
pending_items = initial_items
pending_meta = initial_meta

final_results = {}
final_meta = []

for attempt_index in range(cfg.max_branch_attempts):
    retry_sources = []

    response = await emulate_actions(pending_items)

    for item, meta in zip(pending_items, pending_meta):
        result = response["branch_results"].get(meta[2])

        if _is_resolved_branch_result(result):
            final_results[meta[2]] = result
            final_meta.append(meta)
        elif attempt_index + 1 < cfg.max_branch_attempts:
            retry_sources.append((item, meta, result))
        else:
            final_results[meta[2]] = result
            final_meta.append(meta)

    if not retry_sources:
        break

    pending_items, pending_meta = make_retry_items(retry_sources)
```

実コードでは既存の以下の仕組みを維持します。

- `max_batch_size`
- server の batch limit
- search-wide deadline
- `RequestRejectedError`
- branch cleanup
- `stats.emulate_actions_ms`
- `stats.branches_created`

重要なのは、途中 attempt の fault を `final_results` に残さないことです。

---

## 4. retry は fresh `branch_id` を使う

retry 時に再利用する論理情報は以下です。

```text
parent_branch_id
rng_id
decision_point_id
action_id
```

変更するのは `branch_id` です。

例:

### 初回

```python
{
    "parent_branch_id": "root",
    "branch_id": "bs-search-10",
    "rng_id": 7,
    "decision_point_id": "d-root",
    "action_id": "end",
}
```

### retry

```python
{
    "parent_branch_id": "root",
    "branch_id": "bs-search-11",
    "rng_id": 7,
    "decision_point_id": "d-root",
    "action_id": "end",
}
```

retry item の生成 helper を追加します。

```python
def _make_retry_item(
    self,
    item: Mapping[str, Any],
    meta: tuple[BeamNode, CandidateProposal, str, int],
) -> tuple[
    JsonObject,
    tuple[BeamNode, CandidateProposal, str, int],
]:
    node, candidate, _old_branch_id, rng_id = meta

    new_branch_id = self._allocator.next_branch_id()

    retry_item = dict(item)
    retry_item["branch_id"] = new_branch_id

    retry_meta = (
        node,
        candidate,
        new_branch_id,
        rng_id,
    )

    return retry_item, retry_meta
```

### RNG 方針

retry は原則として同じ `rng_id` を維持します。

目的は「別の stochastic sample を取ること」ではなく、「同一 logical candidate / RNG hypothesis の branch execution を再試行すること」だからです。

ただし実装前に、RL 側で fresh `branch_id` に対して同一 `rng_id` を再使用できることを確認します。

---

## 5. `all_branch_ids` には全 attempt を登録

retry で作成した branch も cleanup 対象です。

例えば、

```text
strike attempt 1 -> b10 fault
strike attempt 2 -> b11 fault
strike attempt 3 -> b12 success
```

なら、

```python
all_branch_ids == [
    ...,
    "b10",
    "b11",
    "b12",
]
```

とします。

各 successful `emulate_actions` admission 後に、従来と同様に以下を実行します。

```python
all_branch_ids.extend(meta[2] for meta in chunk_meta)
stats.branches_created += len(chunk_meta)
```

`branches_created` は logical candidate 数ではなく、実際に作成した physical branch attempt 数を表します。

---

## 6. `_score_frontier()` には最終 attempt の結果だけ渡す

Oracle target の意味を壊さないため、これは重要です。

### 回復した場合

```text
attempt 1 -> fault
attempt 2 -> success
```

この場合、`_score_frontier()` には attempt 2 の success だけを渡します。

途中の attempt 1 fault から `BranchFaultTrace` は生成しません。

### 最終的にも失敗した場合

```text
attempt 1 -> fault
attempt 2 -> fault
attempt 3 -> fault
```

この場合、attempt 3 の fault を `_score_frontier()` へ渡します。

その結果、既存の `_record_branch_fault()` から `BranchFaultTrace` が生成されます。

この変更により `BranchFaultTrace` の意味は、

```text
一度でも fault した branch
```

ではなく、

```text
最大 retry 後にも materialize できなかった logical candidate
```

になります。

Oracle は `BranchFaultTrace` が存在すると該当 RNG subtree を censor するため、途中 fault を trace に残さないことが重要です。

---

## 7. retry 用 stats を追加

retry の観測性を維持するため、`BeamSearchStats` に以下を追加することを推奨します。

```python
@dataclass
class BeamSearchStats:
    ...
    branches_created: int = 0
    branches_faulted: int = 0
    branch_retry_faults: int = 0
    branch_retry_recoveries: int = 0
```

意味は以下です。

```text
branches_created
    実際に生成した physical branch attempt 数

branches_faulted
    最大試行回数後にも回復しなかった logical candidate 数

branch_retry_faults
    retry に回された途中 fault 数

branch_retry_recoveries
    1回以上 fault した後に成功した logical candidate 数
```

例:

```text
A: fault -> fault -> success
B: success
C: fault -> fault -> fault
```

の場合:

```text
branches_created        = 7
branches_faulted        = 1
branch_retry_faults     = 4
branch_retry_recoveries = 1
```

既存の runtime summary が `dict(vars(stats))` を使用している場合、新しい stats も自動的に診断出力へ含まれます。

---

## 8. batch-level rejection は retry しない

現在の `RequestRejectedError` 処理は維持します。

```python
except RequestRejectedError as exc:
    ...
    return (
        final_results,
        final_meta,
        f"emulate_actions_rejected:{detail}",
    )
```

つまり責務は以下です。

```text
per-branch status=faulted
    -> branch-local retry

batch status=rejected
    -> immediate fatal / existing handling
```

`stale_decision_point` のような batch-level protocol rejection を branch ID だけ変えて3回繰り返すことはしません。

---

## 9. search-wide deadline は延長しない

retry を追加しても search 全体の deadline は現在のものをそのまま使用します。

```python
remaining = deadline - time.monotonic()
```

したがって、

```text
search timeout 120 sec
```

なら、branch retry も含めて120秒以内です。

以下のようにはしません。

```text
120 sec x 3 attempts
```

時間切れの場合は従来どおり、

```python
return final_results, final_meta, "time_budget"
```

へ遷移します。

---

## 10. `AllBranchesFaultedError` の位置は維持

Search loop 側の以下の判定は基本的に残します。

```python
if (
    emulated_item_meta
    and not next_beam
    and not newly_finished
    and not hit_continuation_limit
):
    ...
    raise AllBranchesFaultedError(
        "all emulate_actions branch results faulted"
    )
```

ただし意味が変わります。

### 変更前

```text
candidate A -> 1回 fault
candidate B -> 1回 fault
candidate C -> 1回 fault

=> AllBranchesFaultedError
```

### 変更後

```text
candidate A -> 最大3回 fault
candidate B -> 最大3回 fault
candidate C -> 最大3回 fault

=> AllBranchesFaultedError
```

つまり、`AllBranchesFaultedError` は「一度の frontier execution の全滅」ではなく、「branch-local retry を使い切った後の logical frontier 全滅」に近い意味になります。

---

## 11. `oracle_collection.py` の whole-search retry を削除

現在の Runner 側の概念的な処理:

```python
consecutive_branch_faults = 0

while True:
    try:
        oracle_result = await self._oracle.collect(...)
        outcome = await self._commit_engine.decide(...)
    except AllBranchesFaultedError:
        consecutive_branch_faults += 1
        if consecutive_branch_faults >= 3:
            decision_aborted = True
            break
        continue
```

を削除します。

変更後:

```python
try:
    oracle_result = await self._oracle.collect(
        instance_id,
        decision,
        timeout_s=oracle_timeout_s,
    )

    outcome = await self._commit_engine.decide(
        instance_id,
        timeout_s=decision_timeout_s,
        decision=decision,
    )

except AllBranchesFaultedError:
    decision_aborted = True
```

これにより責務は以下になります。

```text
BeamSearchEngine
    -> candidate branch retry を担当

OracleEpisodeRunner
    -> 探索自体は1回だけ実行
```

`MAX_CONSECUTIVE_BRANCH_FAULTS = 3` は削除できます。

これを残すと、

```text
branch attempt x 3
whole search x 3
```

となり、同一 candidate を実質最大9回程度試す可能性があります。

---

## 12. termination reason は最初のPRでは互換性維持

既存の、

```text
aborted_repeated_branch_failure
```

は解析コードや既存データから参照されている可能性があります。

そのため最初のPRでは文字列を変更せず、意味だけ更新する方針を推奨します。

```python
termination_reason = "aborted_repeated_branch_failure"
```

必要ならコメントを追加します。

```python
# "repeated" now refers to exhausted per-candidate branch attempts,
# not repeated whole-search executions.
```

termination reason の rename は別PRで行う方が安全です。

---

## 13. `_emulate_depth_batch()` の実装イメージ

```python
async def _emulate_depth_batch(
    self,
    instance_id: str,
    items: Sequence[JsonObject],
    item_meta: Sequence[tuple[BeamNode, CandidateProposal, str, int]],
    all_branch_ids: list[str],
    stats: BeamSearchStats,
    deadline: float,
) -> tuple[
    dict[str, Any],
    list[tuple[BeamNode, CandidateProposal, str, int]],
    str | None,
]:
    cfg = self.config

    batch_size = cfg.max_batch_size
    server_limit = _server_batch_limit(self._client)
    if server_limit is not None:
        batch_size = min(batch_size, server_limit)

    pending_items = list(items)
    pending_meta = list(item_meta)

    final_results: dict[str, Any] = {}
    final_meta: list[
        tuple[BeamNode, CandidateProposal, str, int]
    ] = []

    for attempt_index in range(cfg.max_branch_attempts):
        retry_sources: list[
            tuple[
                JsonObject,
                tuple[BeamNode, CandidateProposal, str, int],
                Any,
            ]
        ] = []

        for start in range(0, len(pending_items), batch_size):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                for _item, meta, result in retry_sources:
                    final_results[meta[2]] = result
                    final_meta.append(meta)

                return final_results, final_meta, "time_budget"

            chunk_items = pending_items[start : start + batch_size]
            chunk_meta = pending_meta[start : start + batch_size]

            t0 = time.monotonic()

            try:
                response = await self._client.emulate_actions(
                    instance_id,
                    chunk_items,
                    timeout_s=remaining,
                    simulation_options=cfg.simulation_options,
                )
            except RequestRejectedError as exc:
                stats.emulate_actions_ms += (
                    time.monotonic() - t0
                ) * 1000.0

                if _client_unusable(self._client):
                    raise

                for _item, meta, result in retry_sources:
                    final_results[meta[2]] = result
                    final_meta.append(meta)

                fault_kind = exc.response.get("fault_kind")
                detail = (
                    fault_kind
                    if isinstance(fault_kind, str) and fault_kind
                    else type(exc).__name__
                )

                return (
                    final_results,
                    final_meta,
                    f"emulate_actions_rejected:{detail}",
                )

            stats.emulate_actions_ms += (
                time.monotonic() - t0
            ) * 1000.0

            all_branch_ids.extend(meta[2] for meta in chunk_meta)
            stats.branches_created += len(chunk_meta)

            chunk_results = response.get("branch_results") or {}

            for item, meta in zip(chunk_items, chunk_meta):
                branch_id = meta[2]
                result = chunk_results.get(branch_id)

                if _is_resolved_branch_result(result):
                    final_results[branch_id] = result
                    final_meta.append(meta)

                    if attempt_index > 0:
                        stats.branch_retry_recoveries += 1

                    continue

                if attempt_index + 1 >= cfg.max_branch_attempts:
                    final_results[branch_id] = result
                    final_meta.append(meta)
                    continue

                retry_sources.append((item, meta, result))

        if not retry_sources:
            break

        stats.branch_retry_faults += len(retry_sources)

        pending_items = []
        pending_meta = []

        for item, meta, _result in retry_sources:
            retry_item, retry_meta = self._make_retry_item(
                item,
                meta,
            )
            pending_items.append(retry_item)
            pending_meta.append(retry_meta)

    return final_results, final_meta, None
```

実装時には deadline 到達や途中 chunk rejection の際に、それ以前に成功した candidate と retry 待ち fault candidate をどう final result として返すかをテストで固定します。

---

## 14. `tests/decision/test_beam_search.py` の変更

現在の fake connection が、

```text
(parent_branch_id, action_id) -> 固定結果
```

であれば、retry sequence を表現できるように拡張します。

例えば:

```python
self.emulate_result_sequences = {
    ("root", "strike"): [
        {
            "status": "faulted",
            "fault_kind": "worker_exception",
            "error": "timeout",
        },
        {
            "status": "completed",
            "decision_point_id": "d-strike",
            "masked_emulator_dto": _victory_dto(),
        },
    ],
}
```

各 `(parent_branch_id, action_id)` の呼び出し回数に応じて次の結果を返します。

### 追加する主要テスト

#### A. 1回 fault、2回目 success

```text
strike: fault -> success
end: success
```

確認項目:

```text
emulate_actions call count = 2
2回目には strike だけが含まれる
end は再送されない
best action を正常に選べる
```

#### B. 2回 fault、3回目 success

```text
strike: fault -> fault -> success
```

確認項目:

```text
3 attempts
fresh branch_id each attempt
same parent_branch_id
same decision_point_id
same action_id
same rng_id
```

#### C. 3回すべて fault

```text
strike: fault -> fault -> fault
```

確認項目:

```text
4回目を送らない
最終 fault のみ _score_frontier() に入る
branches_faulted == 1
```

#### D. sibling success は再試行しない

```text
A: fault -> success
B: success
```

2回目 request は A だけを含むことを確認します。

#### E. cleanup

retry で生成したすべての physical branch ID が cleanup 対象に含まれることを確認します。

#### F. batch rejection

retry round で `RequestRejectedError` が発生した場合に、それ以上 retry しないことを確認します。

#### G. 全 candidate exhausted

全 candidate が最大3回すべて fault の場合だけ `AllBranchesFaultedError` が発生することを確認します。

#### H. time budget

retry 前または retry 中に deadline に到達した場合、追加 request を送らず `time_budget` で終了することを確認します。

---

## 15. `tests/runner/test_oracle_collection.py` の変更

現在 whole-search retry を確認しているテストは新設計と矛盾するため置き換えます。

### 削除・変更対象

概念的に以下を確認する既存テスト:

```text
commit search が2回 faultして3回目で成功
-> commit_engine.attempts == 3
```

または、

```text
3回 fault
-> runner が3回試行して abort
```

は削除または修正します。

### 新しい期待値

`BeamSearchEngine` 内で retry し尽くした結果、`commit_engine.decide()` が `AllBranchesFaultedError` を返した場合:

```python
self.assertEqual(commit_engine.attempts, 1)
```

とします。

Runner は whole-search retry を行わず、その decision を abort します。

---

## 16. Oracle trace について

途中 attempt の fault を既存の `BranchFaultTrace` として記録しない方針にします。

理由:

Oracle target builder は branch fault があると該当 RNG subtree を `no_target` にするためです。

例えば、

```text
attempt 1 -> worker timeout
attempt 2 -> completed
```

で attempt 1 の fault を `BranchFaultTrace` として残すと、正常に回復して結果を得られたにもかかわらず学習対象が censor されます。

そのため:

```text
途中 fault
    -> stats だけに記録

最終 attempt も fault
    -> BranchFaultTrace を生成
```

とします。

将来的に retry attempt を詳細に解析したい場合は、既存 `BranchFaultTrace` の意味を変えずに、別イベントを追加します。

例:

```python
BranchRetryTrace(
    logical_candidate_id=...,
    attempt_index=1,
    branch_id=...,
    fault_kind=...,
    detail=...,
)
```

ただし最初のPRでは scope を増やしすぎないため、stats のみでもよいと考えます。

---

## 17. fault taxonomy の扱い

最初の実装では次の2案があります。

### 案A: per-branch `faulted` はすべて retry

実装が単純です。

```python
status == "faulted" -> retry
```

メリット:

- Issue #75/#76 のような `worker_exception` timeout を確実に retry できる
- fault_kind taxonomy に Training が依存しない

デメリット:

- 決定論的 fault も3回実行する可能性がある

### 案B: retryable fault_kind のみ retry

例:

```python
def _is_retryable_branch_fault(result: Mapping[str, Any]) -> bool:
    fault_kind = result.get("fault_kind")
    detail = str(result.get("error") or "")

    return (
        fault_kind == "task_timeout"
        or (
            fault_kind == "worker_exception"
            and "Timed out waiting for the next decision point or settlement" in detail
        )
    )
```

メリット:

- deterministic fault を無駄に retry しない

デメリット:

- RL 側 fault taxonomy と密結合する
- 未分類の一時 fault を取りこぼす可能性がある

### 推奨

最初のPRでは、

```text
batch rejection -> retryしない
per-branch faulted -> 最大3回 retry
```

までに留めます。

fault-kind selective retry は実運用ログを見て第2段階で追加する方が安全です。

---

## 18. 実装後の責務分離

```text
OracleEpisodeRunner
│
├─ Oracle collect                    <- 1回
│    │
│    └─ BeamSearchEngine
│         │
│         ├─ candidate A
│         │    ├─ attempt 1 fault
│         │    ├─ attempt 2 fault
│         │    └─ attempt 3 success
│         │
│         ├─ candidate B
│         │    └─ attempt 1 success
│         │
│         └─ candidate C
│              ├─ attempt 1 fault
│              ├─ attempt 2 fault
│              └─ attempt 3 fault
│                   └─ final BranchFaultTrace
│
└─ Commit engine                     <- 1回
     │
     └─ BeamSearchEngine
          └─ 同じ branch-local retry semantics
```

責務は以下になります。

```text
BeamSearchEngine
    candidate の物理 branch execution と retry を管理

OracleEpisodeRunner
    episode / decision lifecycle を管理
    search 全体を retry しない
```

---

## 19. 実装順序

安全に進める場合は以下の順序を推奨します。

### Step 1

`BeamSearchConfig.max_branch_attempts` を追加。

### Step 2

`_is_resolved_branch_result()` と `_make_retry_item()` を追加。

### Step 3

`_emulate_depth_batch()` を branch-local retry 方式へ置換。

### Step 4

`tests/decision/test_beam_search.py` に retry sequence 用 fake とテストを追加。

この時点では Runner の whole-search retry を残したままでも構いません。まず BeamSearch 単体の semantics を固定します。

### Step 5

BeamSearch tests が通った後、`oracle_collection.py` の whole-search retry を削除。

### Step 6

`tests/runner/test_oracle_collection.py` を新しい責務分離に合わせて更新。

### Step 7

Oracle collection integration test を実行し、以下を確認。

```text
一時的 branch fault
    -> 同一 search 内で recovery

全 candidate exhausted
    -> AllBranchesFaultedError
    -> Runner は whole-search retry せず abort
```

### Step 8

Issue #75/#76 に近い End Turn timeout fault を fault-injection で再現し、retry recovery を確認。

---

## 20. 完了条件

以下をすべて満たせば実装完了とします。

```text
[ ] candidate fault 時に、その candidate だけ retry される
[ ] 成功した sibling candidate は再送されない
[ ] 最大試行回数は config で制御できる
[ ] retry ごとに fresh branch_id が発行される
[ ] logical parent / action / decision / RNG hypothesis は維持される
[ ] retry branch を含む全 physical branch が cleanup される
[ ] batch rejection は retry されない
[ ] search-wide deadline を超えて retry しない
[ ] 中間 fault は Oracle の BranchFaultTrace に残らない
[ ] retry exhausted の最終 fault は BranchFaultTrace に残る
[ ] 全 logical candidate exhausted 時のみ AllBranchesFaultedError になる
[ ] OracleEpisodeRunner は whole-search retry をしない
[ ] 既存 normal-success BeamSearch の挙動を壊さない
[ ] tests/decision/test_beam_search.py が通る
[ ] tests/runner/test_oracle_collection.py が通る
```

---

## まとめ

今回の変更の中心は `OracleEpisodeRunner` ではなく `BeamSearchEngine._emulate_depth_batch()` です。

変更前:

```text
candidate を各1回実行
    -> frontier 全滅
    -> search 全体を再実行
    -> 最大3 search attempts
```

変更後:

```text
search は1回
    -> faulted candidate だけ再実行
    -> candidate ごと最大3 attempts
    -> それでも全 candidate fault の場合だけ search failure
```

これにより、Issue #75/#76 のような一時的 worker / emulator fault に対して、探索全体を高コストでやり直すのではなく、失敗した branch だけを局所的に再試行できる構造になります。
