@./README.md

## スクリプトを書く際の指針

- CLIは `typer` で実装し、`app = typer.Typer(...)` を定義する
- ファイル冒頭で `PROJECT_DIR = Path(__file__).resolve().parents[2]` を `sys.path` に追加し、`data_produce` 配下を絶対importする
- データの読み書きは `data_produce/util/file_io.py` の `read_csv`/`save_figure`/`save_table` 経由で行う（`data_registry.json` への自動記録を効かせるため）
- ログは `data_produce/util/logging_config.py` の `setup_logger(script_name)` を使う。出力先は自動的にスクリプト自身と同じディレクトリの `logs/{script_name}.log`（呼び出し元スクリプトのディレクトリから解決される）。ログは日本語のまま
- 図中のテキストは英語表記にする（実行環境にCJK対応フォントが入っておらず、日本語だとタイトル等が文字化けするため）
- 出力先は `save_figure`/`save_table` の `outputs_dir` を省略することで、自動的にスクリプト自身と同じディレクトリの `outputs/{player_type}/{year}/` 配下になる（例: `analysis/overview/overview.py` なら `analysis/overview/outputs/{player_type}/{year}/`）
- スクリプトの目的・引数の詳細・実装上の注意はスクリプト自身のdocstring/コメントに書く。実行コマンドと概要は [COMMANDS.md](COMMANDS.md) に追記する
