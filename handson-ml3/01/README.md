# 01_the_machine_learning_landscape

## 概要

https://github.com/ageron/handson-ml3/blob/main/01_the_machine_learning_landscape.ipynb （Hands-On ML 第1章「機械学習の全体像」）に対応する学習フォルダ。

実際に何をしているかの詳細（各スクリプトの役割・使い方）はサブフォルダの README / AGENTS.md を参照:

- [data_produce/AGENTS.md](data_produce/AGENTS.md) — データの取得・前処理（CSV作成）
- [learn/AGENTS.md](learn/AGENTS.md) — ノートブックの再現（写経） + 追加学習
- [experiment/AGENTS.md](experiment/AGENTS.md) — ハイパーパラメータを壊して観察
- [mine/AGENTS.md](mine/AGENTS.md) — 自分のデータ・課題への適用
- [reinvent/linear.py](reinvent/linear.py) — 最小二乗法のスクラッチ実装（README未整備）

ここでは、このフォルダを通じて**何の概念を学んだか / 何がまだ学べていないか**だけをまとめる。

## 学んだ概念

- **モデルベース学習 vs インスタンスベース学習**
  線形回帰（`learn/linear.py`）と k近傍法（`learn/kneighbor.py`）を同じデータに適用し、両者の予測の違いを比較した。
- **過学習・未学習の可視化**
  多項式回帰の次数を上げていくことで過学習が育つ様子を`learn/overfitting_model_plot.py`と`experiment/degree_sweep.py`で確認した。
- **ハイパーパラメータが汎化に与える影響の体感**
  KNNの`k`（`experiment/knn_k_sweep.py`）と多項式回帰の`degree`（`experiment/degree_sweep.py`）を変化させ、過学習・未学習の連続的な変化を観察した。
- **未知データへの手法の適用**
  `mine/apply.py`で、lifesat以外の任意の2列CSVに線形回帰とKNNを当てはめる汎用テンプレートを作った。
- **最小二乗法の内部原理**
  `reinvent/linear.py`でsklearnに頼らず単回帰をスクラッチ実装し、傾き・切片の導出過程とsample_weightによる外れ値の影響緩和を確認した。
- **教師なし学習（次元削減・クラスタリング）**
  `oecd_bli.csv`（42カ国×24指標）を`data_produce/prepare_oecd_bli_wide.py`でwide形式に整形し、`learn/pca_visualize.py`でPCAによる2次元圧縮、`learn/kmeans_clustering.py`でKMeansクラスタリングを行った。`learn/elbow_silhouette.py`でinertia・silhouette_scoreによる「正解ラベルなしでのk選定」を、`learn/dbscan_clustering.py`で密度ベース手法（ノイズ点の検出、KMeansとの違い）を確認した。
- **訓練・検証・テストの分離と評価指標**
  `learn/train_test_evaluate.py`で`train_test_split`によるホールドアウト検証を行い、「全データで学習した見た目の良さ」と「未知データでの実際の性能」の差を`mean_squared_error`・`r2_score`で確認した。`experiment/split_seed_sweep.py`で分割の乱数シード次第で指標がどれだけブレるか（小データ特有の問題）を観察し、`learn/cross_validation.py`の`cross_val_score`（5-fold）でより安定した評価ができることを確認した。`mine/apply.py`にも同じ評価を組み込み、自分のデータに対しても汎化性能を数値で報告できるようにした（`LEARNING_PLAN.md`参照）。

## 未学習の概念・今後望まれる学習

このフォルダで実際に書かれたコードから見て、追加の学習が望まれる観点:

- **半教師あり学習・自己教師あり学習・強化学習**: 教師あり（回帰）・教師なし（クラスタリング・次元削減）は扱ったが、残る学習方式の実例は未実施。
- **バッチ学習 vs オンライン学習**: 今回はすべてバッチ学習。ストリーミングデータに対する`partial_fit`などのオンライン学習は未実施。
- **過学習・未学習以外のデータ品質の課題**: 訓練データの量的不足、非代表的なサンプリング（サンプリングバイアス）、無関係な特徴量の影響は未検証。`reinvent/linear.py`のsample_weightは外れ値への対処に近いが、体系的な実験にはなっていない。
- **データミスマッチ / train-dev set**: 未実施。

これらは本フォルダの`mine`層での追加課題として、または後続フォルダで扱うことが望ましい。
