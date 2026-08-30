# 便利スクリプト

機械学習・統計の学習とは直接関係ない、データ取得等の便利スクリプトを配置。

- `download_kaggle_dataset.py`: `kagglehub` 経由で Kaggle のデータセット/コンペティションデータをダウンロードし `kaggle/data/{dataset-name}/` に配置する
- `download_handson_ml3_originals.py`: `handson-ml3` のオリジナル README とルート直下のノートブックをダウンロードし `handson-ml3/original/` に配置する
- `internal/`: レポジトリ管理等、機械学習の学習に関連しない便利スクリプトを配置。詳細は `internal/AGENTS.md` を参照

## 実行方法

リポジトリルート（`mlTraining/`）から `uv run` で実行する。

```bash
uv run python scripts/download_kaggle_dataset.py titanic --competition
uv run python scripts/download_handson_ml3_originals.py
```

`download_kaggle_dataset.py` の利用には Kaggle 認証が必要。事前に `~/.kaggle/kaggle.json` を配置するか、環境変数 `KAGGLE_USERNAME` / `KAGGLE_KEY` を設定すること。
