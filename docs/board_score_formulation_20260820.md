**Combat Board Score の数式化と再設計**

## 0. 文章の目的

この文書は、`HeuristicValueFunction` の加算型 board score がターン途中の Combat 状態を比較する際に持つ問題を数式で特定し、代替となる `DamageRaceValueFunction` の設計根拠を残す。記述時点の案とその後の実装を区別し、現在の公開契約は [02_decision_core.md](02_decision_core.md) を正本とする。

### 記号

| 記号 | 意味 | DTO |
|---|---|---|
| $H,\;H_{\max}$ | 自分の HP / 最大 HP | `hp`, `maxHp` |
| $b$ | 自分の block | `block` |
| $\varepsilon,\;E$ | 残エネルギー / 最大エネルギー | `energy`, `maxEnergy` |
| $e_i,\;E_i,\;\beta_i$ | 敵 $i$ の HP / 最大 HP / block | `enemies[].hp/maxHp/block` |
| $d_i$ | 敵 $i$ の予告ダメージ（非攻撃 intent なら 0） | `intent.attackDamage × attackRepeats` |
| $I=\sum_i d_i$ | 今ターンの被予告ダメージ合計 | |
| $R=\sum_i e_i$ | 残敵 HP | |
| $n$ | 生存敵数 | |
| $\tau$ | ターン番号 | `turnNumber` |

---

## 1. 概要

### 1. 現行の数式（A1 適用後）

$$
V_{\text{now}}(s)\;=\;
\underbrace{40\cdot\frac{H-\max(0,\,I-b)}{H_{\max}}}_{\text{effective\_hp\_ratio}}
\;-\;\underbrace{30\cdot\frac{R}{\sum_i E_i}}_{\text{enemy\_hp\_ratio}}
\;-\;\underbrace{2n}_{\text{enemies\_alive}}
\;+\;2\bigl(P^{+}-P^{-}\bigr)-2\bigl(Q^{+}-Q^{-}\bigr)+1\cdot N
$$

$P^{\pm}$ は自分の Buff/Debuff スタック数（1 スタック上限 3）、$Q^{\pm}$ は敵側の同様の量、
$N$ は名前付き power スコアだが $\texttt{DEFAULT\_POWER\_VALUES}=\{\}$ のため恒等的に $N=0$。

終端は $V=\pm 10^5$（exact）。

---

### 2. 次元解析 — なぜこの形では成立しないか

### 2.1 単位が状況依存

第1項の係数: $\dfrac{40}{H_{\max}}$ 〔点 / 自HP〕。$H_{\max}=80$ なら **0.5 点/HP**。

第2項の係数: $\dfrac{30}{\sum_i E_i}$ 〔点 / 敵HP〕。

| 敵構成 | $\sum E_i$ | 敵 HP 1 点の価値 | 自 HP 1 点との比 |
|---|---:|---:|---:|
| 雑魚 1 体 | 38 | 0.79 点 | **1.58 倍** |
| 雑魚 3 体 | 114 | 0.26 点 | 0.53 倍 |
| ボス | 250 | 0.12 点 | 0.24 倍 |

**「自 HP 1 と敵 HP 1 の交換レート」が敵の総 HP で 6 倍以上変動する。**
同じ 6 ダメージの Strike が、局面によって「HP 9.5 相当」にも「HP 1.4 相当」にもなる。
これは設計意図ではなく、比率を取ったことの副作用である。

### 2.2 加算モデルは「変換」を表現できない

$V_{\text{now}}$ は独立な特徴量の重み付き和である。しかし戦闘の実体は**変換**である。

- エネルギー → ダメージ or block
- block → 防いだ HP
- 敵 HP の減少 → 残りターン数の減少 → 被弾総量の減少

加算モデルには「$x$ を消費して $y$ を得る操作が価値中立か否か」という概念が無い。
重みを手で合わせても、変換レートが局面依存（§2.1）である限り整合しない。

### 2.3 同点の発生（実測で 76% の誤選択の説明）

$\alpha=40/H_{\max}$ とおく。Defend の block を $\beta$、次ターン予告を $I'$ とする。

**A: Defend → End Turn**

$$V_A=\alpha\Bigl(H-\max(0,I-b-\beta)-\max(0,I')\Bigr)$$

**B: End Turn → Defend**

$$V_B=\alpha\Bigl(H-\max(0,I-b)-\max(0,I'-\beta)\Bigr)$$

$I-b-\beta\ge 0$ かつ $I'-\beta\ge 0$（block を撃ち漏らさない通常ケース）のとき

$$V_A=\alpha\bigl(H-I+b+\beta-I'\bigr)=V_B .$$

**厳密に一致する。** これは A1 が意図して作ったターン不変性の帰結であり、
`max(actionable, key=state_score)` は最初の最大要素を返すので、
順序 = policy 順 → **End Turn が policy 1 位のとき同点を拾う**（実測 48%）。

### 2.4 同点の何が間違っているか

物理的には A と B は等価ではない。leaf 時点で

- A: ターン2の**開始時**。$\varepsilon=E$、手札まるごと未使用
- B: ターン2の**途中**。エネルギーを 1 消費済み

A は「次の予告 $I'$ をまだ防げる」のに、$V_{\text{now}}$ は $-\max(0,I')$ を**満額請求している**。

> **欠陥の正確な記述**: 現行式は、まだ防ぐ資源が残っている状態に対して、
> 次ターンの被弾を全額計上している。「未消費エネルギーを加点していない」ではなく、
> **「未消費エネルギーが持つ防御能力を無視して被弾を確定させている」**。

---

### 3. 概念化

状態を 2 つに分ける。

$$s=(\underbrace{H,\;R,\;\text{powers},\;\text{deck}}_{\text{持続 (durable)}},\;\underbrace{b,\;\varepsilon,\;\text{hand}}_{\text{一時 (ephemeral)}})$$

**一時リソースはターン終了で消滅する。したがってその価値は、消滅前に持続量へ変換できる分に等しい。**

正しい値関数が満たすべき公理:

| # | 公理 | 意味 |
|---|---|---|
| **A1** | **変換中立性** | エネルギーを block/ダメージへ変換する操作は、それ自体では価値を変えない |
| **A2** | **持続量単調性** | $H$ 増、$R$ 減は価値を増やす |
| **A3** | **ターン位置不変性** | 同じ持続量・同じ「まだ出せる分」なら、ターンのどこにいても同値 |
| **A4** | **単位一貫性** | すべての項が同一単位（自 HP）で表される |

現行式は A1（変換の概念が無い）、A3（§2.4）、A4（§2.1）を満たさない。

$V_{\text{now}}$ が A3 を満たすように見えたのは、$b$ と $I$ を打ち消し合わせて
**変換後の量だけを見た**からで、変換**前**の量（$\varepsilon$, hand）は依然として見えていない。

### まとめ

現行の欠陥は「未消費エネルギーを加点していない」ではなく、

> **まだ防ぐ資源が残っている状態に、次ターンの被弾を満額請求していること**

であり、根はスカラー加算モデルが**変換**（エネルギー→防御/攻撃、敵HP→残ターン→被弾）を
表現できないことにある。提案は全項を自 HP 単位に揃え、変換を明示することで
公理 A1〜A4 を満たし、§5 の 3 ケースすべてで正しい順序を与える。

---

## 2. Architecture

### 4. 提案する数式

以下は当時の設計案である。閉形式による最終形（補遺）は `DamageRaceValueFunction` として実装されたが、この節の初期案そのものを現在の契約として扱わない。

**単位は自 HP に統一する。$V$ は「この戦闘が終わったときの HP の見積り」。**

### 4.1 変換レート

| 記号 | 意味 | 求め方 |
|---|---|---|
| $\kappa$ | エネルギー 1 → 敵 HP | 自己校正（§4.4） |
| $\sigma$ | エネルギー 1 → block | キャラ事前分布（Ironclad: Defend = 5） |
| $\rho$ | 1 ターンの与ダメージ | 自己校正（§4.4） |

### 4.2 今ターンの決着

残エネルギーは**まず被弾を防ぐのに使い、余りを攻撃に回す**（貪欲配分）。

$$
\begin{aligned}
\text{need} &= \max(0,\;I-b) && \text{防ぎ切れていない被弾}\\
\varepsilon_b &= \min\!\left(\varepsilon,\;\left\lceil \text{need}/\sigma \right\rceil\right) && \text{防御に回すエネルギー}\\
\text{leak} &= \max(0,\;\text{need}-\varepsilon_b\,\sigma) && \text{それでも通るダメージ}\\
\varepsilon_a &= \varepsilon-\varepsilon_b && \text{攻撃に回るエネルギー}\\
R_{\text{eff}} &= \max\!\left(0,\;\sum_i (e_i+\beta_i)\,v_i - \varepsilon_a \kappa\right) && v_i\ \text{は Vulnerable 等の係数}
\end{aligned}
$$

### 4.3 残りの戦闘

$$
T=\left\lceil \frac{R_{\text{eff}}}{\rho} \right\rceil ,\qquad
L=\max\bigl(0,\;\bar D-E\sigma\bigr)
$$

$T$ は今ターン以降に敵の手番が回る回数、$L$ は 1 ターンあたりの正味 HP 損失
（$\bar D$ は敵の平均予告ダメージ。当面は $I$ を代用）。

$$
\boxed{\;V \;=\; H \;-\; \text{leak} \;-\; T\cdot L\;}
$$

終端は現行どおり exact（ただしスケールは §whole_run_deepening_plan §P1-B で別途見直す）。

### 4.4 自己校正（カード効果量テーブル不要）

カードのダメージ値は DTO にも外部データにも存在しない（実測: `dmg_per_play` は 439 枚
すべて 0.0）。しかし**その戦闘で自分が実際に出している量**は DTO から計算できる。

$$
\rho=\frac{\sum_i E_i-\sum_i e_i}{\max(1,\;\tau-1)},\qquad
\kappa=\frac{\sum_i E_i-\sum_i e_i}{\max\bigl(1,\;E(\tau-1)+(E-\varepsilon)\bigr)}
$$

デッキ・強化状態・キャラに自動追随し、外部データを必要としない。
$\tau=1$ では分母が無いのでキャラ別事前分布にフォールバックする。

---

### 5. 検証（既知の 3 ケースに当てはめる）

$H_{\max}=80,\;E=3,\;\sigma=5,\;I=I'=12$ とする。

### 5.1 §2.3 の同点ケース

初期: $b=0,\;\varepsilon=1$、手札に Defend 1 枚。

| | 現行 $V_{\text{now}}$ | 提案 $V$ |
|---|---|---|
| A: Defend→End Turn（ターン2開始, $H-7$, $\varepsilon=3$） | $\alpha(H-19)$ | need $=12$, $\varepsilon_b=3$, leak $=0$ → $\;H-7-TL$ |
| B: End Turn→Defend（ターン2途中, $H-12$, $b=5$, $\varepsilon=2$） | $\alpha(H-19)$ | need $=7$, $\varepsilon_b=2$, leak $=0$ → $\;H-12-TL$ |

**現行は完全同点。提案は $V_A-V_B=+5$** ＝ ターン1で Defend が実際に防いだ 5 HP。
同点は tiebreak ではなく**実量で解消**される。

### 5.2 「End Turn がエネルギーを回復するから得になる」懸念

ターン開始時（$b=0,\;\varepsilon=3$、手札満杯、$I=12$）:

| 行動 | 提案 $V$ |
|---|---|
| 何もしない（現状態） | need$=12$, $\varepsilon_b=3$, leak$=0$ → $H-TL$ |
| **End Turn** | 被弾 12 → $H-12$、ターン2で $\varepsilon=3$ → leak$=0$ → $\;H-12-TL$ |
| Defend を打つ | $b=5,\varepsilon=2$: need$=7$, $\varepsilon_b=2$, leak$=0$ → $H-TL$（**中立** = 公理 A1） |
| Strike を打つ | $\varepsilon=2$: need$=12$, $\varepsilon_b=2$, leak$=2$、$R_{\text{eff}}$ 減 → $H-2-(T-1)L$ |

**End Turn は $-12$ で明確に不利。** エネルギーを単独で加点していないので、
「End Turn がエネルギー回復で得をする」は起きない。エネルギーは
**防御能力・攻撃能力としてのみ**評価され、End Turn 後の状態も同じ式で評価されるため対称。

Defend が価値中立になるのも正しい（どうせ防ぐ予定の被弾を、先に確定させただけ）。
Strike は 2 HP を払って $R$ を削る取引として現れる。

### 5.3 $I=0$（非攻撃 intent、実測 28.5%）

need $=0$ → $\varepsilon_b=0$ → 全エネルギーが $\varepsilon_a$ に回り $R_{\text{eff}}$ が減る。

- Defend を打つ: block は leak に効かず $R$ も減らない → **価値中立**（正しい。無駄ブロック）
- Strike を打つ: $R_{\text{eff}}$ 減 → $T$ 減 → **価値増**
- End Turn: $\varepsilon_a$ を捨てて次ターンへ → $R_{\text{eff}}$ が減らない分**不利**

現行では 3 つとも同点（$I=0$ なら block も HP も動かない）だったが、
提案では正しく順序が付く。

---

### 6. 現行式との対応

| 現行の項 | 提案での扱い |
|---|---|
| `effective_hp_ratio` | $H-\text{leak}$ に吸収（ただし残エネルギーの防御力込み） |
| `enemy_hp_ratio` | $T=\lceil R_{\text{eff}}/\rho\rceil$ 経由で**自 HP 単位に変換**（§2.1 の単位問題が消える） |
| `enemies_alive` | 不要（$R$ と $\bar D$ に含まれる。Minion は $v_i$ で割引） |
| `buff_debuff_score` | $\rho,\ \bar D,\ \sigma$ への係数へ移す |
| `named_power_score` | 同上（Vulnerable → $v_i$、Strength → $\rho$ or $\bar D$） |

パラメータは $\sigma$（キャラ事前分布）と $\bar D$ の外挿方法だけが手置きで、
$\rho,\kappa$ は自己校正、残りは DTO の実量である。
現行の 6 個の手調整重みより**自由度が減る**。

---

## 3. API

現在の値関数は `sts2_training.decision.damage_race_value.DamageRaceValueFunction` であり、公開する初期化引数と `ValueModel` 契約は [02_decision_core.md](02_decision_core.md) および実装の `src/sts2_training/decision/damage_race_value.py` を参照する。この計画文書は API の正本ではない。

## 4. 使用例

この文書には、現在の API に対して実行可能と検証した単独コマンド例はない。評価用の実行方法は [02_decision_core.md](02_decision_core.md) を参照する。

## 5. 補足説明

### 7. リスクと段階

| リスク | 対処 |
|---|---|
| $\tau=1$ で $\rho,\kappa$ が未定義 | キャラ別事前分布。1 ターン目は現行式にフォールバックしてもよい |
| $\sigma$ が手置き | Ironclad は Defend=5 が支配的。将来はログから校正 |
| $T$ の階段関数で不連続 | $T=R_{\text{eff}}/\rho$ を実数のまま使う（切り上げない） |
| 貪欲配分が最適でない | 探索が実際の手を試すので、値関数は「残りを平均的に使う」近似で足りる |

**段階**

1. ~~§4 の式を `HeuristicValueFunction` の隣に新実装（既存は残す）~~（実装済み: `src/sts2_training/decision/damage_race_value.py` の `DamageRaceValueFunction`）
2. `selection-logs` の全 DTO で両者を再生し、§5 の 3 ケースと
   `whole_run_deepening_plan` §A2 の指標をオフライン比較（エミュレータ不要、数分）
3. オフラインで改善が出たら実戦 n=20〜30

### 補遺: $T\cdot L$ を閉形式で解く

#### A.1 $T\cdot L$ の何が概念的だったか

§4.3 の $V=H-\text{leak}-T\cdot L$ には**内部矛盾**がある。

- 今ターン: 残エネルギーを「まず防御、余りを攻撃」に貪欲配分（§4.2）
- 将来ターン: $L=\max(0,\bar D-E\sigma)$、すなわち**全エネルギーを防御に回す**前提

後者を採ると与ダメージ $\rho=0$、したがって $T=R/\rho=\infty$。式が自己矛盾している。
さらに $T$ と $L$ を独立な因子として掛けているが、**両者は同じエネルギー予算を奪い合う**
（攻撃に回せば $T$ が減り $L$ が増える）。この結合を無視した積は根拠を持たない。

#### A.2 正しい定式化 — 配分を最適化問題として書く

1 ターンのエネルギー予算 $E$ を、攻撃 $x$ と防御 $E-x$ に分ける。

$$
\rho(x)=x\kappa \quad(\text{与ダメージ/ターン}),\qquad
\mu(x)=(E-x)\sigma \quad(\text{軽減/ターン})
$$

残敵 HP $R$ を削り切るまでのターン数と、その間の HP 損失は

$$
T(x)=\frac{R}{x\kappa},\qquad
\ell(x)=\max\bigl(0,\ \bar D-(E-x)\sigma\bigr)+c
$$

$c>0$ は**1 ターンあたりの固定消耗**（削りダメージ、状態異常の蓄積、敵のスケーリング）。
$c$ を入れないと「永久に防御し続ければ無傷」という退化解が生じるので、必須である。

プレイヤーは損失を最小化するので、**残り戦闘の HP 損失は次の最小化問題の値**である。

$$
\mathrm{Loss}(R,\bar D)\;=\;\min_{0<x\le E}\ \frac{R}{x\kappa}\Bigl(\max\bigl(0,\ \bar D-(E-x)\sigma\bigr)+c\Bigr)
$$

#### A.3 閉形式解

被積分関数は $x$ について区分的に単調なので、解析的に解ける。

**領域 1（完全防御可能、$x\le x_0:=E-\bar D/\sigma$）**

$\max(\cdot)=0$ なので $\dfrac{Rc}{x\kappa}$、これは $x$ について減少。よって最大の $x=x_0$ で最小:

$$\mathrm{Loss}_1=\frac{R\,c}{\kappa\,\bigl(E-\bar D/\sigma\bigr)}\qquad (E\sigma>\bar D\ \text{のときのみ有効})$$

**領域 2（$x>x_0$）**

$$\frac{R}{x\kappa}\bigl(\bar D-E\sigma+x\sigma+c\bigr)=\frac{R}{\kappa}\left(\sigma+\frac{\bar D-E\sigma+c}{x}\right)$$

$\bar D-E\sigma+c>0$ なら $x$ について減少なので $x=E$ で最小:

$$\mathrm{Loss}_2=\frac{R\,(\bar D+c)}{E\,\kappa}$$

**統合**

$$
\boxed{\;
\mathrm{Loss}(R,\bar D)=\frac{R}{\kappa}\cdot
\min\!\left(
\frac{c}{\max\bigl(0,\;E-\bar D/\sigma\bigr)},\;\;
\frac{\bar D+c}{E}
\right)\;}
$$

第1項は分母が 0 以下のとき $+\infty$（＝完全防御が不可能）とする。

**連続性の確認**: $\bar D\to E\sigma^-$ で $\mathrm{Loss}_1\to\infty$、$\mathrm{Loss}_2\to \dfrac{R(E\sigma+c)}{E\kappa}$。
min は $\mathrm{Loss}_2$ に切り替わるので**不連続点は無い**。
$\bar D\ll E\sigma$ では $\mathrm{Loss}_1\approx \dfrac{Rc}{E\kappa}<\mathrm{Loss}_2$ となり、正しく「削りだけ」に漸近する。

**$R$ について線形**であることに注意。傾き

$$\frac{\partial\,\mathrm{Loss}}{\partial R}=\frac{1}{\kappa}\min\!\left(\frac{c}{\max(0,E-\bar D/\sigma)},\frac{\bar D+c}{E}\right)
\;=:\;\lambda\quad\text{〔自HP / 敵HP〕}$$

$\lambda$ が **§2.1 で問題にした「自 HP と敵 HP の交換レート」の正体**である。
手で置いた重みではなく、$\kappa,\sigma,\bar D,E,c$ から導出される量になる。

#### A.4 今ターンは近似せず厳密に解く

今ターンだけは $I$、$b$、$\varepsilon$ が既知なので、貪欲配分ではなく同じ最小化を厳密に行う。
防御に回すエネルギーを $u\in[0,\varepsilon]$ とすると

$$
\Phi(u)=\underbrace{\max\bigl(0,\;I-b-u\sigma\bigr)}_{\text{今ターン通る被弾}}
\;+\;\mathrm{Loss}\Bigl(\max\bigl(0,\;R_v-(\varepsilon-u)\kappa\bigr),\ \bar D\Bigr)
$$

$\Phi$ は $u$ の**区分線形関数**なので、最小値は端点か折れ点でとる。候補は 3 つだけ:

$$u\in\Bigl\{\,0,\quad \min\bigl(\varepsilon,\ \max(0,I-b)/\sigma\bigr),\quad \varepsilon \,\Bigr\}$$

（順に「全部攻撃」「ちょうど防ぎ切る」「全部防御」）。3 点を評価して最小を取る。**探索不要、分岐 3 本。**

$R_v=\sum_i (e_i+\beta_i)\,v_i$ は Vulnerable 等の係数 $v_i$ で補正した実効残敵 HP。

#### A.5 最終形

$$
\boxed{\;
V(s)\;=\;H\;-\;\min_{u\in\{0,\,u^\ast,\,\varepsilon\}}\Phi(u)\;}
$$

終端は exact（$\pm$ terminal utility）。$V$ は**「この戦闘が終わったときの HP の見積り」**であり、
単位は自 HP で統一されている。

#### A.6 パラメータ

| 記号 | 意味 | 決め方 |
|---|---|---|
| $\kappa$ | エネルギー1 → 敵HP | **自己校正** $\dfrac{\sum E_i-\sum e_i}{E(\tau-1)+(E-\varepsilon)}$ |
| $\rho$ | — | $\kappa$ に吸収された（$\rho=x\kappa$）ので独立パラメータではない |
| $\sigma$ | エネルギー1 → block | キャラ事前分布（Ironclad: Defend = 5）。将来ログ校正 |
| $\bar D$ | 敵の平均予告ダメージ | 当面 $I$（公開 intent）を代用 |
| $c$ | 1ターンの固定消耗 | 唯一の自由定数。オフライン校正（初期値 1〜2 HP） |
| $v_i$ | 敵ごとの被ダメージ係数 | Vulnerable 1.5、Minion 割引 など |

**手置きは $\sigma$ と $c$ の 2 個**。現行の 6 個の重み（40 / −30 / −2 / 2 / 2 / 1）より自由度が減る。
しかも $\rho$ が消えたことで §4 の素案よりさらに 1 個減っている。

#### A.7 検証（§5 の 3 ケースを新式で）

$H_{\max}=80,\ E=3,\ \sigma=5,\ \kappa=6,\ c=1,\ I=I'=12$。

### (a) §2.3 の同点ケース

| | 状態 | $\Phi$ の最小 | $V$ |
|---|---|---|---|
| A: Defend→End Turn | $H-7$, $b=0$, $\varepsilon=3$ | $u=3$: leak $=0$ → $\mathrm{Loss}(R)$ | $H-7-\mathrm{Loss}(R)$ |
| B: End Turn→Defend | $H-12$, $b=5$, $\varepsilon=2$ | $u=2$: leak $=0$ → $\mathrm{Loss}(R)$ | $H-12-\mathrm{Loss}(R)$ |

$$V_A-V_B=+5$$

もし双方が攻撃側の解を選んだ場合（$u=0$）:
$V_A=H-19-\mathrm{Loss}(R-18)$、$V_B=H-19-\mathrm{Loss}(R-12)$ で、
$\mathrm{Loss}$ は $R$ 単調増加だから **やはり $V_A>V_B$**。
どちらの領域でも正しい順序が出る。

### (b) End Turn がエネルギー回復で得をするか

ターン開始時 $(H,\ b=0,\ \varepsilon=3,\ I=12)$:

| 行動 | $V$ | 差 |
|---|---|---|
| 現状態 | $H-\mathrm{Loss}(R)$ | — |
| **End Turn** | $H-12-\mathrm{Loss}(R)$ | $\mathbf{-12}$ |
| Defend | $b=5,\varepsilon=2$: $u=2$ で leak $=0$ → $H-\mathrm{Loss}(R)$ | $0$（**中立**、公理 A1） |
| Strike | $\varepsilon=2$: $u=2$ で leak $=2$ → $H-2-\mathrm{Loss}(R-6)$ | $-2+6\lambda$ |

**エネルギーは単独では一切加点されない。** End Turn 後の状態も同じ式で評価されるため対称で、
「End Turn がエネルギー回復で得をする」は構造的に起こらない。
Strike は「2 HP を払って敵 HP 6 を削る」取引として現れ、$\lambda>1/3$ なら得と判定される。

### (c) $\bar D=0$（非攻撃 intent、実測 28.5%）

$\mathrm{Loss}_1=\mathrm{Loss}_2=\dfrac{Rc}{E\kappa}$ で一致。被弾は無いがターンは消費する、
という正しい評価になる。この局面では

- Defend: leak も $R$ も動かない → **中立**（無駄ブロック。正しい）
- Strike: $R$ 減 → $\mathrm{Loss}$ 減 → **有利**
- End Turn: $\varepsilon\kappa$ 分の削りを捨てる → **不利**

現行式では 3 つとも同点だった。

#### A.8 退化ケースの扱い

| 条件 | 扱い |
|---|---|
| $\tau=1$（$\kappa$ 未定義） | キャラ別事前分布。$\kappa_0=$ 基本攻撃札のダメージ/コスト |
| $\kappa=0$（まだ 1 点も削れていない） | $\kappa\leftarrow\max(\kappa,\kappa_{\min})$ でクランプ |
| $R_v-\varepsilon\kappa\le 0$ | $\mathrm{Loss}=0$。今ターンで倒し切る |
| $E\sigma\le\bar D$ | 第1項 $=+\infty$、自動的にレース解に落ちる |
| $V<0$ | 死亡予測。`defeat_penalty` へ滑らかに接続 |

#### A.9 実装量

- ~~`HeuristicValueFunction` と同じ `ValueModel` インタフェース、新クラス 1 個~~（実装済み: `src/sts2_training/decision/damage_race_value.py` の `DamageRaceValueFunction`）
- ~~分岐 3 本 + 閉形式 1 本。ループ無し、追加の emulate 呼び出し無し~~（実装済み: `DamageRaceValueFunction._turn_cost()` と `_remaining_loss()`）
- ~~既存の `CombatObservation` に $\varepsilon$、`maxEnergy`、敵 block、intentTypes の読み出しを追加~~（実装済み: `src/sts2_training/decision/combat_observation.py` の `CombatObservation`）

### 実装との相違・未完了事項

- §4.2〜§4.3 の最初の貪欲配分案は実装されていない。実装は補遺 A.2〜A.5 の閉形式を採用する。
- `enemy_effective_hp_multipliers` は `DamageRaceValueFunction` にあるが既定値は空である。Vulnerable や Minion の係数を既定契約にはしていない。
- 段階 2 の selection log 全 DTO によるオフライン比較、および段階 3 の n=20〜30 実戦比較は、このリポジトリのソースとテストから完了を確認できないため未実装の記録として残す。
