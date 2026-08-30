# data_produce — データ取得・前処理層

CSVのダウンロードや加工など、他の層（`learn` / `experiment` / `mine`）が
使うデータを準備するだけの層。学習ロジックはここに書かず、`data/` に
CSVを吐き出すことだけに専念する。

他の層はネットワークにアクセスせず、`data/` 配下のCSVをローカルから
読み込む処理に集中する。CSVの読み書きは `util/` 層が抽象化しているため、
呼び出し側はフォルダ階層を気にせず、CSVファイル名だけで読み書きできる。

## util/ — CSV読み書きの抽象化

`data/` 配下のCSVを読み書きする薄いラッパー（`util/csv_io.py`）。

- `read_csv(csv_name, **kwargs)` — `data/{csv_name}` を読み込む（`pd.read_csv` にそのまま委譲）
- `write_csv(df, csv_name, **kwargs)` — `data/{csv_name}` に書き出す（`DataFrame.to_csv` にそのまま委譲）
- `data_path(csv_name)` — `data/{csv_name}` の絶対パスを返す
- `exists(csv_name)` — `data/{csv_name}` が既に存在するか

`data_produce` 配下のスクリプト（同じディレクトリで実行される）は
`from util.csv_io import read_csv` のように直接importできる。

`learn` / `experiment` / `mine` など他の層から使う場合は、
チャプター直下（`01_the_machine_learning_landscape/`）を `sys.path` に追加してから
`from data_produce.util.csv_io import read_csv` する（各スクリプト冒頭を参照）。

## スクリプト

- `download_csv.py`
    - `lifesat.csv`（閾値内データ）をそのままダウンロードして `data/` に配置する
- `prepare_life_sat_data.py`
    - `oecd_bli.csv` と `gdp_per_capita.csv` をダウンロードし、結合・整形して
      `lifesat.csv`（閾値内データ）と `lifesat_full.csv`（閾値外も含む完全データ）を
      `data/` に生成する

## 出力

- `data/oecd_bli.csv`, `data/gdp_per_capita.csv` — ダウンロードした生データ
- `data/lifesat.csv`, `data/lifesat_full.csv` — 前処理済みデータ（他の層から参照される）
