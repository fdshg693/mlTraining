# 機械学習の勉強

## 学習教材
https://github.com/ageron/handson-ml3/tree/main

原本（README・各ipynbファイル）は`original/`に、Pythonスクリプトへの変換結果は`converted/`にまとめている（取得・変換の手順は [原本の取得・変換](#原本の取得変換) を参照）。

各{ipynb_file_name}.ipynbファイルに対応する、`{ipynb_file_name}/` フォルダを区切って学習していく

## フォルダ構成

ルート直下: `01..`. `02..`のように、各ipynbファイルごとにフォルダを区切る
- 各サブフォルダ配下:
    - `learn`: ipynbファイルをローカルで再現するためのスクリプトを配置する
        - 再現にこだわる必要はなく、簡略化あるいは複雑化する、またノートブックにない概念を学習してもよい
    - `exercise`: `learn` で学んだ内容をもとに、発展・応用的な課題を行う
    - `experiment`: パラメータを壊す・予測ゲーム
    - `mine`: 同じ手法を自分のデータor課題に適用
    - `reinvent`: 既存ライブラリの手法を自分で実装してみる
    - `data_produce`: 後続で利用するデータの取得・作成を行うスクリプト
        - `data` フォルダにデータを配置する

## `{ipynb_file_name}/learn`の対応例

クラウド: https://github.com/ageron/handson-ml3/blob/main/{ipynb_file_name}.ipynb 
ローカル: `{index of ipynb_file_name}/learn`

## 原本の取得・変換

[ageron/handson-ml3](https://github.com/ageron/handson-ml3)のREADMEおよびルート直下の各`*.ipynb`ファイルは、以下の2ステップでローカルに取得・変換する。

1. `python scripts/download_handson_ml3_originals.py`
   - `original/README.md`と、ルート直下の各`*.ipynb`ファイル（`original/{ipynb_file_name}.ipynb`）をダウンロードする
   - 取得対象のipynb一覧はGitHub APIから動的に取得し、APIが使えない場合はスクリプト内の固定リストにフォールバックする
2. `python scripts/internal/conver_notebook.py original/{ipynb_file_name}.ipynb --output converted`
   - `original/{ipynb_file_name}.ipynb`を`converted/{ipynb_file_name}.py`に変換する（コードセルはそのままPythonコード、Markdownセルはコメントとして出力）
   - `--output`には出力先ディレクトリ、または`.py`で終わる出力先ファイルパスを指定できる（省略時は入力ipynbと同じ場所に出力）

`AGENTS.md`にある通り、`*.ipynb`を直接読む代わりに`converted/`配下の対応する`.py`ファイルを参照する。