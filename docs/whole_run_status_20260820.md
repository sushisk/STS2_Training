# Whole Run 評価状況

## 0. 文章の目的

Oracle collect による学習結果を使った Whole Run 評価について、現在の実行条件、これまでに試した設定、到達階層の結果、score ログから確認できた挙動を整理する。

特に、学習済み action score、beam search、learned pruner のどこが性能上のボトルネックになっているかを、後続の実験で比較できる形に残すことを目的とする。

## 1. 現状

以下の`top_k_actions=8`の5戦記録は、2026-08-20に判明したBeam scope不整合の修正前に取得した
比較用記録である。P0修正後の検証結果は、同文書末尾の備考に追記している。

現在の主な評価条件は次のとおり。

- キャラクター: `IRONCLAD`
- Ascension: `0`
- モデル: 学習済み action score
- board score: `heuristic`
- pruner: 学習済み pruner
- search mode: `standard`
- beam depth: `2`
- beam width: `8`
- `top_k_actions`: `8`
- `max_decisions`: `1000`
- score ログ: 有効
- 評価戦数: 5戦

直近5戦の到達階層は `4, 8, 6, 3, 5` だった。

| 指標 | 結果 |
|---|---:|
| 平均到達階層 | 5.2 |
| 分散（母分散） | 2.96 |
| 標準偏差（母標準偏差） | 1.72 |
| 中央値 | 5 |
| 最小 | 3 |
| 最大 | 8 |
| エラー | 0 / 5 |

5戦とも最終的には敗北終了した。`top_k_actions=4` の直前の5戦評価では平均到達階層が5.0だったため、今回の5.2はわずかな上昇に留まっている。ただし、サンプル数が少ないため、改善と断定できる差ではない。

## 2. 試したこととその結果

### 2.1 評価スクリプトの拡張

Whole Run 評価スクリプトに、次の設定を引数で指定できるようにした。

- `--beam-depth`
- `--beam-width`
- `--top-k-actions`
- `--board-score learned|heuristic`
- `--learned-pruner`
- `--detailed-log-dir`（盤面・selection・action score・search trace の収集）

詳細ログを有効にした場合は、rootごとの完全な masked DTO と action score、selection audit、search trace を同一JSONLに記録する。

### 2.2 pruner と board score の確認

学習済み pruner を外し、上位の盤面をそのまま選ぶ条件を調査した。また、board score を `HeuristicValueFunction` に切り替え、pruner は学習済みのままにする条件でも確認した。

これらの調査では、単純に pruner を外すだけで十分な強さになる傾向は確認できなかった。board score を heuristic にしても、カード選択や戦闘中の行動選択に弱さが残った。

### 2.3 `top_k_actions=4` の score ログ解析

直前の5戦ログでは、次の傾向が確認された。

- beam search は190回
- direct action score の記録は199件
- score trace は1478件
- 最終選択が policy の1位候補と一致したのは65回
- 1位候補以外が選ばれたのは125回
- 合法な攻撃行動337件のうち、84件がpolicyのtop-k候補から漏れた
- root候補のうち61/190回で、少なくとも1つの攻撃行動が候補から漏れた
- そのログでは、stable pruner による攻撃行動の除外は確認されず、保持されたノードはすべて `kept=true` だった

具体例として、`STRIKE` 2枚と`BASH`が合法な場面で、policy候補には`POWDERED_DEMISE`、`BASH`、`DEFEND`、`End Turn`が入り、`STRIKE`が候補から漏れていた。その結果、戦闘に有効な攻撃行動がbeam searchの入口に入らないケースがあった。

この時点では、prunerよりも、policyのtop-k候補生成段階での候補漏れが有力な問題候補だった。

### 2.4 `top_k_actions=8` の再評価

同じ評価条件で`top_k_actions`だけを8に変更し、scoreログを有効にして5戦実行した。5戦をまとめて実行すると時間が長くなったため、各戦を1回ずつ分離して実行した。全戦でエラーは発生しなかった。

ログ全体では次の結果だった。

- rootのpolicy proposal: 251回
- 合法行動数: 1382件
- policy候補数: 1371件
- 候補漏れがあったroot: 5/251回
- 候補から漏れた合法カード行動: 11件
- 最終選択がpolicy順位1位だった回数: 93/251回
- 最終選択がpolicy順位1位以外だった回数: 158/251回
- stable prunerのノード: 1003件
- `kept=true`: 1003件
- `kept=false`: 0件

`top_k_actions=8` によって、候補生成時の漏れは`top_k_actions=4`のときより大幅に減った。一方で、最終選択の多くはpolicy順位1位と一致しておらず、beam search後のboard score、継続行動の評価、またはaction score自体の順位付けに別の問題が残っている可能性が高い。

## 3. 備考

- 直近の5戦は、出力先を分けて保存している。
  - 結果JSON: `data/evaluation/whole_run/topk8-retry-00` ～ `topk8-retry-04`
  - scoreログ: `data/evaluation/score_logs/topk8-retry-00` ～ `topk8-retry-04`
- 最初に5戦一括で実行した`topk8-20260820`は実行時間超過となり、5戦分の正式な評価JSONが生成されなかった。そのため、集計結果には含めていない。
- `top_k_actions=8`の5戦は、実行時間の都合で1戦ずつ実行した。乱数seedは各実行で異なる。
- `top_k_actions=4`と`8`の比較は、いずれも5戦のみであり、性能差の統計的な判断には不十分である。
- 現時点では、候補漏れは`top_k_actions=8`でかなり改善しているため、次に調べる優先候補は以下のとおり。
  - action scoreの順位そのものが戦闘上の有効性と一致しているか
  - beam search後のboard scoreが、攻撃・防御・End Turnの選択を適切に評価しているか
  - 深さ2・幅8の探索で、後続行動の評価が十分に反映されているか
  - カード報酬選択および戦闘後の遷移が、Whole Run全体の性能を下げていないか

### P0 scope修正後の検証

- `top_k_actions=8`・1戦: 到達階層9、エラーなし。
- `top_k_actions=4`・5戦: 到達階層`11, 5, 5, 8, 6`、平均7.0、分散5.2、エラーなし。
- いずれも全戦敗北であり、scope counterの正式な受入判定はscoreログ検査が必要。
