# board_eval の使い方

`board_eval` は **Whole Run の状態価値**を扱うための特徴抽出・学習・推論コードです。
Combat 中の戦術評価を担う `decision.value.ValueModel` / `HeuristicValueFunction` /
`CombatDecisionEngine` とは責務を分離しています。`RunStateValueModel` は
`decision.value.ValueModel` を継承せず、このモジュールの値を Combat のスコアへ加算する
仕組みもありません。

## カード特徴

`CardFeatureExtractor.from_csv()` は、リポジトリ内の
`tools/output/card_secondary_features.csv` を読み込みます。デフォルトパスは CWD ではなく
`card_features.py` の位置からリポジトリ内の固定位置を解決します。

```python
from sts2_training.board_eval import CardFeatureExtractor

extractor = CardFeatureExtractor.from_csv()
features = extractor.extract("STRIKE_IRONCLAD", upgraded=True)
vector = features.to_vector()  # FEATURE_NAMES 順
```

upgrade はカード定義の数値差分を推測しません。`CardFeatures.upgrade` はデッキ内の各カード
インスタンスが upgrade 済みかだけを保持し、Deck Summary 側で枚数・比率へ集計します。
upgrade 後の damage/block 等の実数値差分は扱いません。

Energy と Stars は別 resource として扱います。CSV の `star_cost` は
`CardFeatures.star_cost` に保持し、`uses_star_cost` で Star-cost card を識別します。
Energy 0 / Stars N のカードを通常の free 0-Energy card と同じ cost feature にはしません。

従来の単一 `other_effect_magnitude` は廃止し、異なる単位を混ぜないよう次の effect family に
分割しています。

- `heal_total`, `max_hp_gain_total`, `gold_gain_total`
- `card_generation_count`, `card_transform_count`, `card_tutor_count`
- `buff_apply_total`, `debuff_apply_total`
- `orb_generation_count`, `orb_evoke_count`
- `character_resource_gain`
- `other_known_effect_count`, `unparsed_effect_count`

生成・変換・Tutor・Orb 生成の CSV count に `?` が付く場合、その数字は確定値へ加算しません。
`card_generation_uncertain_count`, `card_transform_uncertain_count`,
`card_tutor_uncertain_count`, `orb_generation_uncertain_count` に「count が不確定な effect entry 数」
として別に残します。例えば `SOUL:1?` は `card_generation_count += 1` とはせず、
`card_generation_uncertain_count += 1` とします。

`character_resource_gain` は forge / stars / summon の単純合算です。この PR ではこの契約を
変更しません。

### multi-hit

multi-hit は、回数から `effective_total_damage` のような推測値を作りません。固定 `RepeatVar`
を持つカードは従来どおり `is_multi_hit=True` として扱います。

回数が Run/Combat 状態を参照して動的に決まるカードでは、`CardFeatures.hit_count_reference`
に **何を参照して回数を決めるか**を `ReferenceScaling` として保持します。例えば、
Barrage の Orb 数参照は `COMBAT_COUNT / ORB_COUNT` です。永続 CSV の source snippet が
途中で切れていて参照先を完全には確定できない場合は、確認できる粒度までだけ構造化し、
それ以上は推測しません。未知の式は `UNPARSED` として source snippet を保持します。

`hit_count_reference` 自体は数値 feature ではなく構造メタデータです。現在の線形モデルでは
`is_multi_hit` が numeric feature のままで、参照式からヒット数や実効ダメージを推定しません。

## 参照型カード

参照・スケーリング効果は `ReferenceScaling` として構造を保持します。

```python
@dataclass(frozen=True)
class ReferenceScaling:
    source: ScalingSource
    scope: Scope
    filter_value: str
    referent: Literal["self", "enemy"] | None = None
    coefficient: float | None = None
```

従来の自由文字列 `detail` は廃止しました。例えば enemy の Vulnerable 参照なら
`referent="enemy"`, `filter_value="VULNERABLE_POWER"` です。Exhaust pile 参照なら
`referent=None`, `filter_value="EXHAUST"` です。

## Deck Summary

`summarize_deck()` は type/cost/damage/block/draw 等に加え、次を集計します。

- `upgraded_count`, `upgrade_ratio`
- `upgraded_attack_count`, `upgraded_skill_count`, `upgraded_power_count`
- `unknown_card_count`, `known_card_ratio`
- `curse_count`, `status_count`, `unplayable_count`, `excluded_card_count`
- Energy cost bins と、別 dimension の `star_cost_card_count`, `total_star_cost`, `star_cost_*_count`
- effect family ごとの deck-level total と uncertainty count
- `dynamic_card_count`

Curse / Status / Unplayable / `deck_eval_excluded` のカードは deck size には残しますが、通常の
cost / damage / block / energy efficiency 等の mechanical aggregate には混ぜません。

`damage_per_energy` は damage を持つ **Energy-cost card** の damage と Energy cost だけで計算し、
`block_per_energy` も block contributor だけで計算します。pure block card を追加しただけで
`damage_per_energy` が下がる、または pure attack を追加しただけで `block_per_energy` が下がる
定義にはしません。Star-cost card は total damage/block には残しますが、Energy efficiency の
分子・分母には入れず、Star cost feature で別に表現します。

未知カードを `on_unknown_card="skip"` で処理する場合でも、その枚数は捨てず
`unknown_card_count` と `known_card_ratio` に残します。coverage 閾値による推論拒否や
conservative fallback は実装していません。

## DTO adapter と state_kind

DTO の wire field 名を知るのは `dto_adapter.py` に集約しています。
`board_context_from_dto()` は HP / Max HP / Gold / Floor とともに `state_kind` を返します。
DTO-only の呼び出しでは `currentRoomType` を優先し、なければ DTO 内 `boundary` を fallback に
使います。self-play selection log から `build_examples_from_log()` を作る場合は event-level
`boundary` も渡し、優先順位を **`currentRoomType` → event-level `boundary` → DTO 内 `boundary`**
とします。これにより `currentRoomType` がない decision point でもログ上の boundary を
`state_kind` として保持できます。

`build_examples_from_log(..., state_kinds=...)` で decision point を種類別に絞れます。
event-level `boundary` から得た `state_kind` にも同じ filter が適用されます。デフォルトは
`None` で、現状互換の「フィルタなし」です。どの `state_kind` を Whole Run 学習対象にするか
という最終リストはまだ policy として固定していません。

## Missing value

`hp`, `max_hp`, `gold`, `act_floor`, `total_floor` は、各値の直後に `*_missing` feature を持ちます。
欠損値は数値側を `0.0`、missing flag を `1.0` にします。実値 `0` は数値 `0.0`、missing flag
`0.0` なので、学習時と推論時の両方で「欠損」と「本当に 0」を区別できます。

`MODEL_FEATURE_NAMES` が学習と推論の共通 feature-order 契約です。

## 学習

```sh
python tools/train_board_eval.py \
  --log-dir data/self_play \
  --output tools/output/board_eval_model_weights.json
```

Train / Validation / Test は **Run ファイル単位**で分割し、さらに terminal Win/Lose label ごとに
stratify します。小規模 split が単一クラスになった場合、Log Loss と ROC-AUC は `null` とし、
理由をログへ残します。

必要なら `--state-kind Shop --state-kind RestSite` のように繰り返し指定して学習対象を絞れます。
指定した scope は weights artifact の `training_state_kinds` に保存します。フィルタなしの場合は
`null` です。

weights JSON には係数だけでなく、次の参照用 metadata を保存します。

- `model_type`, `artifact_schema_version`, `created_at`
- `card_catalog_hash`, `card_catalog_path`
- `training_commit`, `training_state_kinds`
- `feature_schema_version`, `feature_schema_hash`

catalog hash/version の厳密な runtime 検証は行いません。metadata は再現性・追跡用です。

## 推論: RunStateValueModel

学習済み線形モデルは `LinearRunStateValueModel` で読み込みます。

```python
from sts2_training.board_eval import LinearRunStateValueModel

model = LinearRunStateValueModel.from_weights_file(
    "tools/output/board_eval_model_weights.json"
)
probability = model.predict_win_probability(masked_emulator_dto)
```

`predict_win_probability()` は `sigmoid(intercept + Σ coef * standardized_feature)` による
`(0, 1)` の Run 勝率推定値を返します。推論側は標準ライブラリのみで動作します。

このモデルを `CombatDecisionEngine(value_fn=...)` へ渡す用途ではありません。Combat の盤面評価は
既存 `decision.value.ValueModel` 系の責務として独立したままです。

## 次実装項目

次に明文化して残す実装項目は **Enchantment を CardInstance の状態として扱うこと**です。

- `has_enchantment`
- `enchantment_type`
- Deck Summary の `enchanted_count` 等

同じ `card_id` / upgrade 状態でも Enchantment によりカード挙動が変わるため、card definition と
card instance の境界を整理した上で実装します。この項目以外の今回見送った提案は、この文書の
次実装リストには追加しません。
