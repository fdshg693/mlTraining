# MLB のデータを使って機械学習

baseballsavant.mlb.com 等から取得したMLBの実データを使い、`handson-ml3` で学んだ手法を自分のデータに適用して学習する。

## フォルダ構成

- `data_produce/`: 後続で使うデータの取得・作成を行うスクリプト。詳細は `data_produce/README.md` / `data_produce/CLAUDE.md` を参照
    - `data/` にデータを配置する
- `analysis/`: 取得したデータの分析・可視化
- `data_registry.json`: `data_produce/util/metadata.py` が自動生成する、データファイルの読み書き元（どのスクリプトが作り、どのスクリプトが読むか）の一覧。手動で編集しない
- `logs/`・`outputs/`: 各スクリプトのログ・生成物は、リポジトリ直下にはまとめず `data_produce/logs/`・`analysis/{module}/logs/`・`analysis/{module}/outputs/` のように各スクリプトと同じディレクトリ配下に置く（`data_produce/util/logging_config.py`・`file_io.py` が呼び出し元スクリプトの場所から自動的に解決する）

## 実行方法

リポジトリルート（`mlTraining/`）から `uv run` で実行する。

```bash
uv run python mlb/data_produce/fetch_leaderboard.py --year 2026 --type pitcher --field ff_avg_speed
```
