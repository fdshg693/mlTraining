@./README.md

- データファイルの読み書きは `data_produce/util/file_io.py` の `read_csv`/`write_csv` 等を経由すること。`data_registry.json` への自動記録が効かなくなるため、`pandas` を直接使って読み書きしない
- `data_registry.json` は自動生成物。手動で編集しない
- `data_produce/` 配下の作業は `data_produce/CLAUDE.md` も参照すること
- `data/`配下のCSVを複数列combineして使う（特に球種別フィールドを含む）場合は、値の意味を誤解しやすい注意点を [data_produce/data/NOTES.md](data_produce/data/NOTES.md) にまとめてあるので必ず確認する。新しい注意点を見つけたら追記する
