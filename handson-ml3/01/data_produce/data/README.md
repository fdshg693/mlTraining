# data/ 配下のファイル一覧と取得方法

このフォルダのCSVはすべて `data_produce/` 配下のスクリプトが生成した
成果物であり、手動で編集しない。再生成したい場合は該当スクリプトを
再実行する（生データはローカルに存在すればダウンロードをスキップする）。

## 生データ・ダウンロード

### gdp_per_capita.csv

- 取得元: https://github.com/ageron/data/raw/main/lifesat/gdp_per_capita.csv
- 取得スクリプト: [`../prepare_life_sat_data.py`](../prepare_life_sat_data.py)
  （`exists()` チェックで未取得時のみダウンロード。`prepare_oecd_bli_wide.py` はこのファイルを使わない）
- 内容: 国別・年別のGDP per capita（Code, Year, GDP per capita (USD) 等の列を含む生データ）
- 加工なし。そのまま保存されたもの

### oecd_bli.csv

- 取得元: https://github.com/ageron/data/raw/main/lifesat/oecd_bli.csv
- 取得スクリプト: [`../prepare_life_sat_data.py`](../prepare_life_sat_data.py) または
  [`../prepare_oecd_bli_wide.py`](../prepare_oecd_bli_wide.py)（どちらも `exists()` チェックで未取得時のみダウンロード）
- 内容: OECD Better Life Index の Country / Indicator / INEQUALITY / Value のlong format生データ
- 加工なし。そのまま保存されたもの

## 加工データ

### lifesat.csv（前処理済み・閾値内データ）

- 生成スクリプト: [`../prepare_life_sat_data.py`](../prepare_life_sat_data.py)
- 生成元: `oecd_bli.csv` + `gdp_per_capita.csv`
- 加工内容:
  1. `gdp_per_capita.csv` から2020年分を抽出し、`Code`/`Year` 列を削除、
     列名を `Country`, `GDP per capita (USD)` に変更
  2. `oecd_bli.csv` を `INEQUALITY == "TOT"`（不平等区分は全体のみ）で絞り込み、
     `Country` を行、`Indicator` を列に持つwide形式にpivot
  3. 上記2つを `Country` で結合し、`GDP per capita (USD)` でソート
  4. `GDP per capita (USD)` 列と `Life satisfaction` 列だけを残す
  5. GDP per capitaが `23,500〜62,500` の範囲内の国のみ抽出（元の書籍と同じ閾値）
- ※ `download_csv.py` を使えば、上記の再現の代わりに
  https://github.com/ageron/data/raw/main/lifesat/lifesat.csv から直接ダウンロードすることも可能
  （ageron本人が生成した完成品をそのまま取得する方法）

### lifesat_full.csv（前処理済み・閾値外を含む完全データ）

- 生成スクリプト: [`../prepare_life_sat_data.py`](../prepare_life_sat_data.py)
- 生成元・加工内容: `lifesat.csv` と同じ手順の1〜3まで
  （4のGDP per capita閾値によるフィルタは適用しない、全41ヶ国分のデータ）

### oecd_bli_wide.csv（前処理済み・クラスタリング学習用）

- 生成スクリプト: [`../prepare_oecd_bli_wide.py`](../prepare_oecd_bli_wide.py)
- 生成元: `oecd_bli.csv`
- 加工内容:
  1. `INEQUALITY == "TOT"` で絞り込み、`Country` を行、`Indicator`（24個）を列に持つwide形式にpivot
  2. `OECD - Total`（個別の国ではなく全体集計値）の行を除外
  3. 欠損値は列（Indicator）ごとの中央値で補完
     （dropnaすると41ヶ国中25ヶ国が失われるため。中央値は外れ値の影響を受けにくい）
- 用途: PCA/KMeans/DBSCANなど教師なし学習の学習データ（`LEARNING_PLAN.md` 参照）

