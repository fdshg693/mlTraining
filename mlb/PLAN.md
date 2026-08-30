# 題材案（handson-ml3 01/02 → mlb データへの応用）

[handson-ml3/01/README.md](../handson-ml3/01/README.md)・[handson-ml3/02/README.md](../handson-ml3/02/README.md)の「学んだ概念」のうち、`mlb/analysis/`（`overview`/`correlation`/`prediction`/`role`/`clustering`）ではまだ扱っていないものを棚卸しし、MLBデータでの題材案としてまとめる。

## 前提: 現状のmlbで実施済みのこと

01/02の概念のうち、mlb側で既に対応済みのもの（再掲不要）:

- モデルベース学習 vs インスタンスベース学習（線形回帰 vs KNN）→ `prediction/knn_compare.py`
- 教師なし学習（PCA・KMeans・elbow/silhouette・DBSCAN）→ `clustering/cluster.py`・`clustering/outliers.py`
- train/test分離と評価指標（`train_test_split`・シードのブレ・`cross_val_score`・MSE/R2）→ `prediction/generalization.py`
- Pipeline/ColumnTransformerによる前処理統合 → `prediction/compare_models.py`
- 複数モデル比較+交差検証（線形回帰 vs RandomForest） → `prediction/compare_models.py`
- ハイパーパラメータ探索（`RandomizedSearchCV`） → `prediction/compare_models.py`
- 特徴量重要度・最終評価・モデル永続化（`joblib`） → `prediction/compare_models.py`
- データ取得を`data_produce/`へ分離（02の未学習指摘の解消） → `data_produce/fetch_leaderboard.py`
- 代表性を保った分割（層化サンプリング。単純`train_test_split`と`StratifiedShuffleSplit`のカテゴリ比率のズレ・R^2のブレ幅を比較） → `prediction/generalization.py`

## 未実施の概念（今回の棚卸し対象）

| # | 概念 | 出典 | 現状 |
|---|------|------|------|
| 3 | 分布変化・データミスマッチ（train-dev/test） | 01の未学習・02の未学習 | 実施済み。`prediction/year_shift.py` |
| 4 | 特徴量重要度の裏付け（Permutation Importance / Partial Dependence） | 02の未学習 | 実施済み。`compare_models.py`に追加 |
| 5 | 外れ値・上限打ち切りの体系的な扱い | 02の未学習 | 部分実施。`clustering/outliers.py`はDBSCANでの検出のみで、除去/キャッピング/目的変数変換の効果比較は未実施 |
| 6 | スケーリングと分布変換（対数変換・RBF類似度特徴量・`TransformedTargetRegressor`） | 02 | 未実施 |
| 7 | `SelectFromModel`による特徴量選択 | 02 | 未実施 |
| 8 | 推論用CLI/APIへの接続 | 02の未学習 | 未実施。`compare_models.py`が保存する`joblib`モデルを読み込んで使う入り口がない |
| 9 | バッチ学習 vs オンライン学習（`partial_fit`） | 01の未学習 | 未実施。**データ制約あり**（下記参照） |
| 10 | 半教師あり学習 | 01の未学習 | 未実施（優先度低・stretch） |

対象外とするもの:
- **最小二乗法のスクラッチ実装（01の`reinvent`層）**: `mlb/analysis/README.md`の「フォルダ分けの基準」は分析観点ごとの構成であり、handson-ml3のような`learn`/`reinvent`層構造を取っていないため、mlbには馴染まない。
- **自己教師あり学習・強化学習（01の未学習）**: リーダーボード形式の集計データとは相性が悪く、別教材で扱うのが適切。
- **カスタム変換器のスクラッチ実装（02の`StandardScalerClone`等）**: 上記reinvent同様、mlbの目的（実データへの適用・分析観点の追加）からは外れるため対象外。

## 題材案（優先度順）

### 5. 外れ値の体系的な扱い — 対応: #5

`clustering/outliers.py`で検出済みのDBSCAN外れ値投手について、(a) 除去、(b) 上限打ち切り（パーセンタイルでのキャッピング）、(c) 目的変数の対数変換、の3パターンで`compare_models.py`のRMSEがどう変わるかを比較する新規スクリプト。`clustering/`と`prediction/`のどちらの観点にも跨るため、`prediction/`側に「外れ値処理ありの回帰」として追加するのが妥当（`outliers.py`の出力CSVを読み込む形）。

### 6. スケーリングと分布変換の比較 — 対応: #6

`compare_models.py`の前処理（`StandardScaler`のみ）に、右に裾の長いフィールド（例: 登板数・イニング数）への対数変換、`rbf_kernel`によるクラスタ中心との類似度特徴量、`TransformedTargetRegressor`によるERAの対数変換を追加し、RMSEの変化を比較する。`compare_models.py`への機能追加。

### 7. `SelectFromModel`による特徴量選択 — 対応: #7

`compare_models.py`のRandomForestに`SelectFromModel`を挿入し、`threshold`を変えたときの選択特徴量数とRMSEの関係を確認する。「特徴量を減らせば必ず性能が上がるわけではない」ことをmlbデータでも確認できる。`compare_models.py`への機能追加、または`prediction/`に新規スクリプト。

### 8. 推論用CLI — 対応: #8

`compare_models.py`が保存する`joblib`モデル（`RandomForestRegressor`＋前処理込み）を読み込み、任意の投手1人分の特徴量（コマンドライン引数、またはCSV1行）からERAを予測する`prediction/predict_from_model.py`を追加する。`predict.py`（単回帰の予測）とは別に、複数特徴量モデルの推論エントリポイントとして位置づける。

### 9. バッチ学習 vs オンライン学習（データ制約あり）— 対応: #9

現状の`data_produce/fetch_leaderboard.py`はシーズン集計のリーダーボードのみを取得しており、試合単位・日付単位のデータソースがない。そのため`partial_fit`によるオンライン学習を体感するには、まず`data_produce/`側で日次・試合単位のデータ取得を追加する必要がある（baseballsavantに該当エンドポイントがあるか要調査）。優先度は低いが、実施する場合は「シーズンを疑似的な時系列とみなし、月ごとに`SGDRegressor.partial_fit`で逐次更新→全データ一括学習と比較」という構成が考えられる。

### 10. 半教師あり学習の体験（stretch）— 対応: #10

`role.py`のルールベース先発/リリーフ区分のうち、一部の投手だけラベルがあるという想定で残りを隠し、`clustering/cluster.py`のクラスタリング結果からラベルを伝播（多数決）させて正解率を見る。教師あり（`role.py`）と教師なし（`clustering/`）の橋渡しとして位置づけられるが、優先度は低い。

## 次のアクション

上記のうち、着手する題材案の番号をこのファイルに残し、実装したら該当セクションを削除して`mlb/analysis/README.md`・`COMMANDS.md`・対応する`{folder}/README.md`（学んだ概念）を更新する運用とする（`stat_catalog.md`が過去の題材案3・4を参照していたのと同じ運用）。
