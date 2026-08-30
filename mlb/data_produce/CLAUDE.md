# data_produce

baseballsavant.mlb.com のリーダーボードCSVを取得し、後続の `learn`/`experiment`/`mine`/`reinvent` から使えるデータとして `data/` 配下に保存するモジュール。

## 使い方

リポジトリルートから実行する。

```bash
uv run python mlb/data_produce/fetch_leaderboard.py \
  --year 2026 --type pitcher --field ff_avg_speed
```

- `--type` / `-t`: `pitcher` か `batter`
- `--field` / `-f`: `savant_client.SavantField` のフィールド。複数指定可（`-f ff_avg_speed -f ff_avg_spin`）。同じ組み合わせは常に同じファイル名になる（後述）
- `--min-sample`: 最小サンプル数（デフォルト `q` = 規定投球回/打席数）。`savant_client.MinSample`のいずれか（`q`, `1`, `10`, `20`, `40`, `100`）
- `--refresh`: 既存CSVがあっても再取得する（デフォルトはキャッシュ利用）

オプション一覧は `--help` で確認できる。

## キャッシュの考え方

シーズン中は値が変動し続けるため、「一度取れば終わり」ではない。
- 既存の `{stat_key}.csv` があればデフォルトでは再取得せず使い回す
- `--refresh` を付けたときだけ再取得する
- `{stat_key}.meta.json` に取得条件（URL・パラメータ・取得日時）を必ず残す。キャッシュヒット時は更新しない＝そのCSVが実際に取得された時点の条件を保つ

保存先: `data/leaderboard/{player_type}/{year}/{stat_key}.csv`（+ 同名の `.meta.json`）。`stat_key` は指定した `--field` の組み合わせから `savant_client.stat_key_for()` が自動生成する（値をソートして連結するため、指定順によらず同じ組み合わせは同じファイル名になる）。

## 新しいフィールドを追加するには

`savant_client.py` の `SavantField` Enumに `メンバー名 = "savantの列名"` を追記するだけ。組み合わせごとの登録は不要（`--field` の指定内容に応じてファイル名は自動生成される）。呼び出し元（CLI）は変更不要。

追加できるフィールドの候補（`savant`の`selections`に指定できる値）は `fetch_stat_catalog.py` で一覧取得できる。

```bash
uv run python mlb/data_produce/fetch_stat_catalog.py --year 2026 --type pitcher
```

baseballsavantのカスタムリーダーボード画面（カラム選択UI）のHTMLを取得し、「Standard Stats」「Statcast Stats」の各見出し配下にあるチェックボックス（`id`がそのまま`selections`の値になる）をラベル付きで列挙し、`data/stat_catalog/{player_type}/{year}/stat_catalog.yaml`（+ 同名の`.meta.json`）へ保存する。キャッシュの考え方は上記のリーダーボードCSVと同様（既存YAMLがあれば再取得せず使い回し、`--refresh`で明示的に再取得）。詳細は [`mlb/analysis/stat_catalog.md`](../analysis/stat_catalog.md) を参照。

## 補助ユーティリティ（`util/`）

- `file_io.py`: `data/` 配下のCSV読み書き・`outputs/` への図表/テーブル保存を薄くラップ。呼び出し側はファイル名だけ指定すればよい
- `metadata.py`: `read_csv`/`write_csv`/`save_figure` などが呼ばれるたびに、対象ファイルと呼び出し元スクリプトの対応をリポジトリルートの `data_registry.json` へ自動記録する。ソースを読まなくても「どのファイルが誰に作られ、誰に読まれているか」を一覧できる
- `logging_config.py`: `setup_logger(script_name)` でスクリプトごとのログを、呼び出し元スクリプトと同じディレクトリの `logs/{script_name}.log`（例: `data_produce/logs/fetch_leaderboard.log`）に出力する

新しいデータ取得スクリプトを追加する際は、`file_io.read_csv`/`write_csv` 経由でファイルを読み書きすること（`data_registry.json` への自動記録を効かせるため）。
