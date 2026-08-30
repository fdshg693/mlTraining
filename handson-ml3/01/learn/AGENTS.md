# 01_the_machine_learning_landscape

## 概要
https://github.com/ageron/handson-ml3/blob/main/01_the_machine_learning_landscape.ipynb を参考に、https://github.com/ageron/data/blob/main/lifesat/lifesat.csv のデータを線形解析を主にして分析・描画する

CSVのダウンロード・前処理は `../data_produce` 層が担当する。ここのスクリプトは
`../data_produce/data` に置かれたローカルCSVを読み込む処理に専念する。

## 写経

- `kneighbor.py`
    - k近傍法を用いて、GDP per capita から Life satisfaction の予測を行う
- `linear.py`
    - 線形回帰を用いて、GDP per capita から Life satisfaction の予測を行う
    - また、閾値内のデータと閾値外のデータを含めた完全なデータセットを用いて、線形回帰の結果を比較する
- `overfitting_model_plot.py`
    - 10次多項式を含めて過学習を示すためのモデルを作成する
- `plot.py`
    - `../data_produce/data` のCSVファイルを読み込み、GDP per capita と Life satisfaction の散布図を描画する

### 訓練・検証・テストの分離と評価指標

ノートブックには薄いホールドアウト検証・交差検証を、`lifesat.csv`を使って学ぶスクリプト群

- `train_test_evaluate.py`
    - `train_test_split`で訓練/テストを分離し、線形回帰・KNNをそれぞれ訓練データだけで学習、
      テストデータで`mean_squared_error`・`r2_score`を算出する。「全データで学習した見た目の
      良さ」と「未知データでの実際の性能」の差を確認する
- `cross_validation.py`
    - `cross_val_score`（5-fold）で、1回のホールドアウトより安定した評価ができることを確認する。
      foldごとのスコアと、平均・標準偏差を出力する

### 教師なし学習

`oecd_bli_wide.csv`（42カ国 × 24指標）を使い、教師あり学習には無い
「正解ラベルなしで構造を発見する」を体感するスクリプト群。

- `pca_visualize.py`
    - 標準化した24指標をPCAで2次元に圧縮し、国を散布図にプロットする
- `kmeans_clustering.py`
    - 標準化した指標にKMeans（k=3固定）を適用し、国をクラスタに分ける。PCAの2次元平面上に色分けして可視化する
- `elbow_silhouette.py`
    - KMeansのkを2〜10で振り、inertia（エルボー法）とsilhouette_scoreを比較してkの選び方を学ぶ
- `dbscan_clustering.py`
    - 同じデータにDBSCANを適用する。クラスタ数を指定せず密度でクラスタを発見し、ノイズ点（外れ値）を検出する。KMeansと違い高次元では密度差が出にくく、パラメータ（eps, min_samples）に敏感な点も確認できる