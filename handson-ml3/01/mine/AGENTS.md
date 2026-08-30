# mine — 自分の問題に適用する層

`learn`（写経）→ `experiment`（壊して観察）で身につけた手法を、
**自分の好きなデータ**に適用する層。ML エンジニアに必要な
「未知のデータに手法を当てる」力はここで伸びる。

## 使い方

1. `data/` に2列の CSV を置く（1列目=特徴量 x、2列目=目的変数 y）
   - 例: ゲームのプレイ時間 vs スコア、勉強時間 vs テスト点、身長 vs 体重 など
   - とりあえず動かすだけなら、デフォルトで data_produce 層の lifesat データを読む
2. `apply.py` の `FEATURE_COL` / `TARGET_COL` を自分の列名に変える
3. 実行して、線形回帰と KNN の予測・散布図を見る

## スクリプト

- `apply.py`
    - 任意の2列 CSV に対して、線形回帰と KNN を当てて比較・描画する汎用テンプレ
    - lifesat 以外のデータでも `FEATURE_COL` / `TARGET_COL` / `CSV_PATH` を変えるだけで動く
    - `train_test_split`で訓練/テストを分離し、訓練R^2だけでなくテストデータでの
      `mean_squared_error`・`r2_score`（＝汎化性能）も出力する（`../LEARNING_PLAN.md`参照）

## 次の一歩（2章以降に向けて）

- 結果をリポジトリ直下の結果ログに1行追記する
