# analysis スクリプト一覧

各スクリプトの実行方法と、何をするCLIかをまとめる。引数の詳細やアルゴリズムの意図はスクリプト冒頭のdocstring、または `--help` を参照。

すべてリポジトリルート（`mlTraining/`）から `uv run` で実行する。対象フィールドのデータを事前に `data_produce/fetch_leaderboard.py` で取得しておく必要がある。

## overview/overview.py

取得済みリーダーボードCSVについて、統計サマリをログへ出力し、分布図とランキング図をPNGで保存する。

```bash
uv run python mlb/analysis/overview/overview.py \
  --year 2026 --type pitcher --field ff_avg_speed
```

## correlation/correlation.py

取得済みリーダーボードCSVについて、2つのフィールド間の相関係数を算出し、散布図（回帰直線つき）をPNGで保存する。

```bash
uv run python mlb/data_produce/fetch_leaderboard.py \
  --year 2026 --type pitcher --field ff_avg_speed --field ff_avg_spin

uv run python mlb/analysis/correlation/correlation.py \
  --year 2026 --type pitcher --x-field ff_avg_speed --y-field ff_avg_spin
```

## prediction/predict.py

取得済みリーダーボードCSVについて、2つのフィールドに単回帰をあてはめ、片方の実測値からもう片方を予測する。信頼区間・予測区間つきの散布図（PNG）と予測値一覧（CSV）を保存する。

```bash
uv run python mlb/data_produce/fetch_leaderboard.py \
  --year 2026 --type pitcher --field ff_avg_speed --field ff_avg_spin

uv run python mlb/analysis/prediction/predict.py \
  --year 2026 --type pitcher --x-field ff_avg_speed --y-field ff_avg_spin --speed 96.5
```

## prediction/generalization.py

取得済みリーダーボードCSVについて、単回帰の汎化性能を可視化する。全データ学習（見た目の良さ）・ホールドアウト（乱数シード違いを複数試行したブレ）・k分割交差検証（安定化）のR^2を箱ひげ図で比較し、表（CSV）も保存する。

加えて、単純`train_test_split`と`StratifiedShuffleSplit`（層化サンプリング）を比較する。層化キーは`--stratify-by`で選び、`target`（目的変数を`pd.qcut`で等頻度カテゴリに分割、デフォルト）と`role`（`role.py`と同じ先発/リリーフ区分）の2種類。mlbの投手データは母数が少なく、単純分割だと層化キーのカテゴリ比率がシード次第で大きくズレるため、層化分割がそのズレをどれだけ抑えるか（と、R^2のブレ幅への影響）を箱ひげ図・表（CSV）で確認できる。

```bash
uv run python mlb/data_produce/fetch_leaderboard.py \
  --year 2026 --type pitcher --field ff_avg_speed --field ff_avg_spin

uv run python mlb/analysis/prediction/generalization.py \
  --year 2026 --type pitcher --x-field ff_avg_speed --y-field ff_avg_spin

# --stratify-by role の場合は p_game / p_formatted_ip も含めて取得しておく
uv run python mlb/data_produce/fetch_leaderboard.py \
  --year 2026 --type pitcher --field ff_avg_speed --field ff_avg_spin \
  --field p_formatted_ip --field p_game --min-sample 40

uv run python mlb/analysis/prediction/generalization.py \
  --year 2026 --type pitcher --x-field ff_avg_speed --y-field ff_avg_spin --stratify-by role
```

## prediction/knn_compare.py

取得済みリーダーボードCSVについて、同じ2フィールドに線形回帰とKNN（`KNeighborsRegressor`、複数の`k`を指定可）を当てはめ、予測直線・予測曲線を重ねた散布図（PNG）と、xグリッド上の予測値一覧（CSV）を保存する。`k`を変えたときの過学習（k小）・未学習（k大）の違いを線形回帰と比較しながら観察できる。

```bash
uv run python mlb/data_produce/fetch_leaderboard.py \
  --year 2026 --type pitcher --field ff_avg_speed --field ff_avg_spin

uv run python mlb/analysis/prediction/knn_compare.py \
  --year 2026 --type pitcher --x-field ff_avg_speed --y-field ff_avg_spin \
  --k 1 --k 5 --k 15 --k 40 --target 96.5
```

## prediction/poly_compare.py

取得済みリーダーボードCSVについて、同じ2フィールドに`PolynomialFeatures`の次数（`degree`、複数指定可）を変えた多項式回帰を訓練データのみで学習し、予測曲線を重ねた散布図（PNG、訓練/テストの点を色分け）と、次数ごとの訓練/テストRMSE表（CSV）を保存する。次数を上げると訓練RMSEは下がり続ける一方、テストRMSEはある次数を境に反転して上がり始める（過学習）様子を確認できる。

```bash
uv run python mlb/data_produce/fetch_leaderboard.py \
  --year 2026 --type pitcher --field ff_avg_speed --field ff_avg_spin

uv run python mlb/analysis/prediction/poly_compare.py \
  --year 2026 --type pitcher --x-field ff_avg_speed --y-field ff_avg_spin \
  --degree 1 --degree 3 --degree 8 --degree 15
```

## prediction/compare_models.py

取得済みリーダーボードCSVについて、複数の投球フィールド（球速・回転数・球種使用率・イニング数・登板数など）を特徴量に、目的変数（デフォルトはERA）を予測する回帰を行う。`ColumnTransformer`で標準化をまとめ、線形回帰と`RandomForestRegressor`を`cross_val_score`（RMSE）で比較する箱ひげ図（PNG）・表（CSV）を保存し、続けて`RandomizedSearchCV`でrandom_forestのハイパーパラメータ（`n_estimators`/`max_depth`/`max_features`/`min_samples_leaf`）を探索、探索結果の表（CSV）を保存する。探索後のモデルの特徴量重要度（不純度ベース、MDI）を棒グラフ（PNG）・表（CSV）で保存する。

続けて、MDIと`sklearn.inspection.permutation_importance`（ホールドアウトで列をシャッフルした際のRMSE悪化）を比較する（`--importance-test-size`でホールドアウト割合、`--n-repeats`でシャッフル回数を指定）。両者を正規化したshareで並べた棒グラフ（PNG）・順位差つきの表（CSV）を保存し、「分岐に使われた頻度」と「実際の性能寄与」がどれだけ一致・不一致するかを確認できる。さらにpermutation importance上位`--pdp-top-n`件について`PartialDependenceDisplay`で特徴量の値と予測値の関係を図（PNG）で保存する。最後にモデルを`joblib`で保存・再読込して予測が一致することを確認する。

```bash
uv run python mlb/data_produce/fetch_leaderboard.py --year 2026 --type pitcher \
  --field ff_avg_speed --field ff_avg_spin --field p_formatted_ip --field p_game \
  --field n_ff_formatted --field n_si_formatted --field n_sl_formatted \
  --field n_cu_formatted --field n_ch_formatted --field n_fc_formatted \
  --field p_era --min-sample 40

uv run python mlb/analysis/prediction/compare_models.py --year 2026 --type pitcher
```

## prediction/year_shift.py

`compare_models.py`と同じ特徴量セット（球速・回転数・球種使用率・イニング数・登板数など）でERAを予測する回帰について、複数年（train-dev、例: 2023-2024年）のデータをまとめて学習した際の`cross_val_score`RMSE（同一分布内の汎化性能の見積もり）と、train-devでは一度も見ていない別の年（test、例: 2025年）に対するRMSE（未知の分布＝年への汎化性能）を、線形回帰とrandom_forestそれぞれについて箱ひげ図（PNG）・表（CSV）で比較する。「年をまたぐと精度がどれだけ落ちるか（分布シフトのコスト）」を確認できる。

```bash
uv run python mlb/data_produce/fetch_leaderboard.py --year 2023 --type pitcher \
  --field ff_avg_speed --field ff_avg_spin --field p_formatted_ip --field p_game \
  --field n_ff_formatted --field n_si_formatted --field n_sl_formatted \
  --field n_cu_formatted --field n_ch_formatted --field n_fc_formatted \
  --field p_era --min-sample 40

uv run python mlb/data_produce/fetch_leaderboard.py --year 2024 --type pitcher \
  --field ff_avg_speed --field ff_avg_spin --field p_formatted_ip --field p_game \
  --field n_ff_formatted --field n_si_formatted --field n_sl_formatted \
  --field n_cu_formatted --field n_ch_formatted --field n_fc_formatted \
  --field p_era --min-sample 40

uv run python mlb/data_produce/fetch_leaderboard.py --year 2025 --type pitcher \
  --field ff_avg_speed --field ff_avg_spin --field p_formatted_ip --field p_game \
  --field n_ff_formatted --field n_si_formatted --field n_sl_formatted \
  --field n_cu_formatted --field n_ch_formatted --field n_fc_formatted \
  --field p_era --min-sample 40

uv run python mlb/analysis/prediction/year_shift.py \
  --train-year 2023 --train-year 2024 --test-year 2025 --type pitcher
```

## role/role.py

取得済みリーダーボードCSVについて、投手を先発/リリーフに分類し、球速に差があるかを箱ひげ図とWelchのt検定で調べる。

```bash
uv run python mlb/data_produce/fetch_leaderboard.py \
  --year 2026 --type pitcher --field p_game --field p_formatted_ip --field ff_avg_speed --min-sample 40

uv run python mlb/analysis/role/role.py --year 2026 --velocity-field ff_avg_speed
```

## clustering/cluster.py

取得済みリーダーボードCSVについて、球種別使用率6種（4シーム・シンカー・スライダー・カーブ・チェンジアップ・カッター）+4シーム球速・回転数を`StandardScaler`→`KMeans`でクラスタリングする。kは`--k`未指定ならinertia（エルボー法）とsilhouette_scoreからk-min〜k-max（デフォルト2〜8）の範囲で自動選定する。`PCA`で2次元に落とした散布図を保存し、role.pyと同じ基準の先発/リリーフ区分を点の形（○/△）で重ねてクラスタとの一致・不一致を見る。

```bash
uv run python mlb/data_produce/fetch_leaderboard.py --year 2026 --type pitcher \
  --field p_game --field p_formatted_ip \
  --field n_ff_formatted --field n_si_formatted --field n_sl_formatted \
  --field n_cu_formatted --field n_ch_formatted --field n_fc_formatted \
  --field ff_avg_speed --field ff_avg_spin --min-sample 40

uv run python mlb/analysis/clustering/cluster.py --year 2026
```

## clustering/outliers.py

`cluster.py`と同じ特徴量・同じCSVに`DBSCAN`を適用し、どのクラスタのコア点の近傍（`--eps`以内）にも属さない「典型的でない投手（外れ値）」を検出する。`eps`/`min-samples`は自動選定できないデータ依存のパラメータのため、ログのクラスタ数・ノイズ点数を見ながら調整する。

```bash
uv run python mlb/analysis/clustering/outliers.py --year 2026 --eps 2.0 --min-samples 5
```
