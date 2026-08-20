# Combat Board Score 再設計案

`HeuristicValueFunction` を既存設計から離れて設計し直す。入力として実際に何が得られるかを
実測で確定し、出力が何であるべきかを消費側から定義し、その間を埋める。

## 1. 出力は何であるべきか

`ValueModel` の出力を消費するのは 3 箇所ある。

| 消費者 | 必要な性質 |
|---|---|
| Beam の leaf 比較 → root action 決定 | **1 探索内の全 leaf に対する順序**。leaf はターン内の位置も残り敵数も異なる |
| `StableFrontierPruner` | 同上（frontier の順序） |
| Oracle bootstrap ラベル | 上記に加えて**学習に使えるスケール**（§whole_run_deepening_plan §1.8） |

3 つに共通する要求はひとつだけである。

> **ターン位置・ターン数・残り敵数が違う状態どうしを、同じ尺度で並べられること。**

この 1 行が、これまでの失敗（block 二重計上 → A1、ターン境界の半手ずれ → A1/A3）の共通原因である。
特徴量を足し合わせる形にしている限り、「今ターンまだ使えるリソース」と「恒久的な状態」が
混ざり、比較は破綻する。

### 目的量

Whole Run の目的は到達階層である。1 つの戦闘が Run に対して持つ意味は

> **その戦闘が終わったときの HP と、そもそも勝てるかどうか**

だけである。block も energy も手札も敵の残り HP も、それを予測する限りにおいてのみ価値を持つ。
したがって理想の出力は

```
V(state) = E[戦闘終了時の HP]  −  大きな罰 × P(この戦闘で死ぬ)
```

これは**定義からしてターン位置に不変**である。「残りの戦闘」を推定する量なので、
ターンのどこで測っても同じ意味になる。現行の特徴量加算にはこの性質が無い。

## 2. 入力として実際に得られるもの（実測）

`data/evaluation/selection-logs` の実 DTO を集計した結果。

### 2.1 使えるもの

| 区分 | フィールド | 備考 |
|---|---|---|
| 恒久 | `hp` / `maxHp` | |
| 恒久 | `enemies[].hp` / `maxHp` / `isAlive` | 死亡した敵も配列に残る（hp=0） |
| 恒久 | `playerPowers[]` / `enemies[].powers[]` | `id` 付き。実測語彙は §2.3 |
| 恒久 | `relics[]` | `id` / `stackCount`。**現行は完全に未使用** |
| 恒久 | `deck` / `drawPile` / `discardPile` / `exhaustPile` | カードは `id` / `type` / `cost` / `upgraded` |
| 一時 | `block`（自分）、`enemies[].block` | **敵 block は現行未使用** |
| 一時 | `energy` / `maxEnergy` | **現行未使用** |
| 一時 | `hand[]` | **現行未使用** |
| 一時 | `potions[]` | 3 スロット、未所持は `null` |
| ターン位置 | `turnNumber` = `combatRoundNumber` | 全 Combat DTO に存在 |
| 脅威 | `enemies[].intent` | §2.2 |

### 2.2 intent は「攻撃かどうか」だけではない（21,264 件の実測）

| `intentTypes` | 件数 | 割合 |
|---|---:|---:|
| `Attack` | 15,207 | 71.5% |
| `StatusCard` | 4,975 | 23.4% |
| `Buff` | 2,237 | 10.5% |
| `Defend` | 649 | 3.1% |
| `Debuff` | 516 | 2.4% |
| `DebuffStrong` | 129 | 0.6% |
| `Summon` | 50 | 0.2% |
| `CardDebuff` | 43 | 0.2% |

`attackDamage` を持つのは `Attack` の 15,207 件のみで、**残り 6,057 件（28.5%）には
ダメージ欄が無い**。現行の `effective_hp` はそこを `incoming_damage = 0` と読むので、
「完全に安全」と評価する。実際には `StatusCard`（デッキに Slimed 等が入る）が 23%、
`Buff`（敵が強化される）が 10% ある。**戦闘中の脅威の 4 分の 1 以上が値に反映されていない。**

### 2.3 power は名前で意味が違うのに、名前が使われていない

実測語彙（上位）:

- 自分: `STRENGTH_POWER`, `RUPTURE_POWER`, `CRIMSON_MANTLE_POWER`, `CONSTRICT_POWER`,
  `SHRINK_POWER`, `FRAIL_POWER`, `VICIOUS_POWER`, `MAYHEM_POWER`
- 敵: `VULNERABLE_POWER`（4,782）, `STRENGTH_POWER`（2,723）, `SLIPPERY_POWER`,
  `ILLUSION_POWER`, `MINION_POWER`, `ARTIFACT_POWER`

現行は `type`（Buff/Debuff）だけを見て ±1 点/スタック（3 で頭打ち）とし、
`DEFAULT_POWER_VALUES` は**空の辞書**である。つまり:

- 敵に付けた `VULNERABLE`（被ダメージ増）が、その他あらゆる debuff と同じ 1 点
- 敵の `STRENGTH`（以後の全攻撃が増える）が、一時的な debuff と同じ 1 点
- `MINION_POWER` の敵も、倒すべき本体と同じ 1 体としてカウント

### 2.4 得られないもの: カードの効果量

**これが設計上いちばん効く制約である。**

- DTO のカードは `id` / `type` / `cost` / `upgraded` のみ。**ダメージ値もブロック値も無い**
- `data/external_data/cards_all.json` には `dmg_per_play` / `blk_per_play` /
  `dmg_per_energy` という欄があるが、**439 枚すべて 0.0**。スキーマだけで中身が無い。
  さらに実戦で手札に来た 17 種のうち 5 種（`SLIMED`, `DAZED`, `GUILTY`, `DOWSING`,
  `NEOWS_FURY`）はファイルに存在しない

したがって「手札からどれだけダメージが出せるか」を**カード表から引くことはできない**。
後述の設計はこれを前提に、**自己校正**で回避する。

## 3. 現行設計が構造的に取りこぼしているもの

| # | 取りこぼし | 影響 |
|---|---|---|
| 1 | 未消費の `energy` / `hand` を評価しない | **A3 が消せなかった半手ずれの正体**（後述 §4.3） |
| 2 | 非攻撃 intent（28.5%）を「脅威ゼロ」と読む | StatusCard / Buff / Summon が見えない |
| 3 | 名前付き power が無効（`DEFAULT_POWER_VALUES = {}`） | Vulnerable も Strength も一般 debuff 扱い |
| 4 | 敵の `block` を無視 | 与ダメージ推定が過大 |
| 5 | `relics` を無視 | 恒久的な戦力差が値に出ない |
| 6 | 予告 1 ターン分しか見ない | 「あと何ターンで倒せるか」が入らない |

## 4. 提案: 残り戦闘のダメージレース残差モデル

### 4.1 形

出力を「予想戦闘終了時 HP」に一本化する。

```
remaining_enemy_hp = Σ effective_hp(enemy)          # vulnerable/block/minion 補正込み
our_dpt            = 自己校正した 1 ターンあたり与ダメージ
turns_to_win       = ceil(remaining_enemy_hp / our_dpt)

their_dpt          = 公開 intent（次の 1 ターンは確定）→ 以降は観測平均
mitigation         = block + 未消費リソースで作れる分

expected_hp_loss   = Σ_{t < turns_to_win} max(0, their_dpt(t) − mitigation(t))
V                  = hp − expected_hp_loss
```

`V ≤ 0` は「このまま行くと死ぬ」を意味し、そこから `defeat_penalty` に滑らかに落とす。
現行の勝利/敗北の exact terminal 値はそのまま残す。

### 4.2 自己校正（カード表なしで `our_dpt` を得る）

カード効果量は無いが、**その戦闘で自分が実際に出しているダメージは DTO から分かる**。

```
damage_dealt_so_far = Σ(enemy.maxHp) − Σ(enemy.hp)      # 死亡敵も配列に残るので総和で取れる
turns_elapsed       = max(1, turnNumber − 1)
our_dpt             = damage_dealt_so_far / turns_elapsed
```

デッキにも強化状態にもキャラにも自動で追随し、外部データを一切必要としない。
戦闘 1 ターン目は分母が無いので事前分布（キャラ別定数）にフォールバックする。

同じやり方で**エネルギー単価**も出せる。

```
energy_spent ≈ maxEnergy × (turnNumber − 1) + (maxEnergy − energy)
dmg_per_energy = damage_dealt_so_far / max(1, energy_spent)
```

### 4.3 未消費リソースの扱い ― A3 の残差はここで消える

A3（leaf のターン整列）が失敗したのは、`turnNumber` を揃えても
**「そのターン内で何手消化したか」が揃わない**ためだった（§whole_run_deepening_plan §10.6）。

| leaf | ターン | そのターンの消化 |
|---|---|---|
| `[カード, カード]` +整列 | 2 | 0 手（開始時） |
| `[End Turn, カード]` | 2 | **1 手消化済み** |

値関数が**未消費エネルギーを、それが将来生む効果として計上**すれば、この差は静的に消える。

```
remaining_enemy_hp −= energy × dmg_per_energy
```

- 「エネルギー 3 を残したターン開始時」と「エネルギー 3 を使って敵 HP を減らした状態」が
  **同じ値になる**
- 追加の emulate 呼び出しは **ゼロ**。A3 のような quiescence 展開が要らない

つまり A3 が探索の追加で解こうとした問題を、**値の定義で解く**。

### 4.4 intent を型で扱う

| `intentTypes` | 扱い |
|---|---|
| `Attack` | `attackDamage × attackRepeats`（現行どおり） |
| `StatusCard` | デッキ希釈の恒久コスト。以後の `our_dpt` を割り引く |
| `Buff` | 敵の以後の `their_dpt` を割り増す |
| `Debuff` / `DebuffStrong` / `CardDebuff` | 自分の以後の `our_dpt` を割り引く |
| `Summon` | `remaining_enemy_hp` の増加を見込む |
| `Defend` | 敵 block の増加 → `turns_to_win` が伸びる |

「今ターンのダメージ」ではなく「**残りターン数と 1 ターンあたりの交換レート**」に効かせるのが要点。
非攻撃 intent が値に入るようになる。

### 4.5 power と relic

named power を**交換レートへの係数**として扱う（点数の加算ではなく）。

- 敵の `VULNERABLE_POWER` → その敵の `effective_hp` を割り引く（こちらの dpt が上がるのと等価）
- 敵の `STRENGTH_POWER` → その敵の `their_dpt` を上げる
- 自分の `STRENGTH_POWER` → `our_dpt` を上げる
- 自分の `FRAIL_POWER` / `SHRINK_POWER` → `mitigation` / `our_dpt` を下げる
- `MINION_POWER` → `remaining_enemy_hp` への寄与を割り引く（本体ではない）
- `relics` は当面 `our_dpt` / `mitigation` の事前分布に効かせるのみ（優先度低）

## 5. 段階的な導入案

一度に全部入れると測定できないので 3 段階に分ける。**各段階を独立に A/B する。**

### Stage 1 — 追加データ不要、現行構造のまま（小）

1. `intentTypes` を読む。非攻撃 intent に脅威値を与える
2. `DEFAULT_POWER_VALUES` を実測語彙で埋める（`VULNERABLE_POWER`, `STRENGTH_POWER`,
   `FRAIL_POWER`, `MINION_POWER` ほか §2.3 の 14 種）
3. 敵の `block` を与ダメージ推定から差し引く
4. **未消費エネルギーを敵 HP の割引として計上**（§4.3）— A3 の残差に直接効く

4 は単独でも A3 の目的を果たす可能性がある。**Stage 1 の中でもこれを最初に単独で測る**べき。

### Stage 2 — 自己校正ダメージレース（中）

`our_dpt` / `their_dpt` / `turns_to_win` を導入し、出力を「予想戦闘終了時 HP」に置き換える。
`DEFAULT_WEIGHTS` の加算モデルからの置き換えになるので、Stage 1 の効果を測ってから。

### Stage 3 — カード効果量テーブル（大・任意）

自前ログ（`selection-logs` の branch 遷移）から `card_id → 中央値ダメージ/ブロック` を
学習して `dmg_per_energy` の精度を上げる。Stage 2 の自己校正で足りるなら不要。

> 注: 本調査で branch 遷移からの効果量抽出を試したが、ログの前後状態の対応付けが
> 期待どおりに取れず（前状態と後状態が同一に見える）、**未検証**である。
> Stage 3 に進む場合はまずログ構造の確認から。

## 6. 検証方法（エミュレータ不要のオフライン A/B）

到達階層は 1 条件 n=10 で CI ±2.8 と粗く、値関数の細かい比較には向かない。
`selection-logs` には**全 branch の DTO が入っている**ので、エミュレータを動かさずに
「同じ探索の leaf を新しい値関数で並べ替えたら選択が変わるか」を再生できる。

見るべき指標（`whole_run_deepening_plan` §A2 と同じ）:

1. `End Turn` が best root で、かつ打てるカードが手札に残っている件数
2. 深さ1 の カード と `End Turn` の平均 value 差
3. ターン位置が違う leaf どうしの値の一貫性（同一局面をターン開始時／ターン途中で
   評価して差が縮むか）

オフラインで 1 と 2 が改善してから実戦 n=20〜30 に進む。これなら 1 変更あたり数分で回る。

## 7. まとめ

| | 現行 | 提案 |
|---|---|---|
| 出力の意味 | 特徴量の重み付き和 | 予想戦闘終了時 HP |
| ターン位置不変性 | 無い（A1 で部分的に改善、A3 で失敗） | 定義から不変 |
| 未消費エネルギー/手札 | 無視 | 敵 HP の割引として計上 |
| 非攻撃 intent（28.5%） | 脅威ゼロ | 型ごとに交換レートへ反映 |
| 名前付き power | 無効（空辞書） | 交換レートの係数 |
| 敵 block | 無視 | 与ダメージから差し引き |
| カード効果量 | 不要 | 不要（自己校正） |
| 追加の emulate 呼び出し | ― | **ゼロ** |

最小の一歩は **§5 Stage 1 の項目 4（未消費エネルギーの計上）単独**である。
実装は小さく、A3 が探索の追加で解こうとした問題を追加コストゼロで解ける可能性がある。
