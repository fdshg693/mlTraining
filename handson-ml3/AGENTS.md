@./README.md

`*.ipynb` ファイルを直接読まないこと!
必ず`original/{name}.ipynb`に対応する`converted/{name}.py`ファイルを読み、存在しない場合は `scripts/internal/conver_notebook.py` を使って作成すること