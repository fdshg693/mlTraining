# Kaggle

Kaggle のデータセット・コンペティションのデータを使った学習用フォルダ。

## フォルダ構成

- `data/`: `scripts/download_kaggle_dataset.py` でダウンロードしたデータを配置
    - `data/{dataset-name}/`: 各データセット/コンペティションのファイル一式
- `titanic/`: Titanic データセットを使った学習用のノートブック・スクリプト

## データ取得

リポジトリルート（`mlTraining/`）から以下を実行し、`kaggle/data/{dataset-name}/` にダウンロードする。

```bash
uv run python scripts/download_kaggle_dataset.py titanic --competition
```

- 通常のデータセットの場合は `--dataset`（デフォルト）、コンペティションのデータの場合は `--competition` を指定する
- 配置先フォルダ名は slug の末尾から自動生成されるが、`--name` で指定も可能
- 事前に Kaggle 認証が必要。`~/.kaggle/kaggle.json` を配置するか、環境変数 `KAGGLE_USERNAME` / `KAGGLE_KEY` を設定すること
