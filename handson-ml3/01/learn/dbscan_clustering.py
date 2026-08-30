# 標準化した24指標（oecd_bli_wide.csv）にDBSCANを適用するスクリプト
#
# ノートブックには無い教師なし学習の体験用スクリプト
# KMeansと違い、クラスタ数kを指定せず密度でクラスタを発見する。
# 密度が低い領域にある点は「ノイズ（外れ値、label=-1）」として扱われる。
# KMeansの結果（kmeans_clustering.py）と可視化を見比べて手法の違いを確認する

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_produce.util.csv_io import read_csv
from data_produce.util.logging_config import setup_logger

logger = setup_logger(Path(__file__).stem)

oecd_bli_wide = read_csv("oecd_bli_wide.csv", index_col="Country")
X = StandardScaler().fit_transform(oecd_bli_wide.values)

# eps: 「近傍」とみなす最大距離、min_samples: コア点とみなすために近傍に必要な点数
# どちらも決め打ちの値。小さすぎるとほぼ全点がノイズになり、大きすぎると1クラスタに潰れる
# （elbow_silhouette.pyのkと違い、DBSCANにはこの2つを自動で選ぶ標準的な指標が無い）
EPS = 4.0
MIN_SAMPLES = 3
dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
labels = dbscan.fit_predict(X)

# labels は各点に割り当てられたクラスタ番号（0, 1, 2, ...）の配列。
# DBSCANは「コア点から辿ってeps以内で繋がる点」をクラスタとしてまとめるが、
# どのコア点のeps近傍にも入らなかった点は、どのクラスタにも属さない「ノイズ」として
# 特別に-1というラベルが割り振られる（0始まりのクラスタ番号と衝突しない値として-1を使う規約）。
# そのため set(labels) には {0, 1, ..., -1} のように -1 が混ざり得るので、
# 実際のクラスタ数を数えるときは -1 を1つ分差し引く。
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)  # ノイズ点（どのクラスタにも属さなかった点）の数
logger.info(f"eps={EPS}, min_samples={MIN_SAMPLES}")
logger.info(f"クラスタ数: {n_clusters}, ノイズ点数: {n_noise}")
for cluster_id in sorted(set(labels)):
    countries = oecd_bli_wide.index[labels == cluster_id].tolist()
    name = "noise" if cluster_id == -1 else f"cluster {cluster_id}"
    logger.info(f"{name}: {countries}")

# 可視化専用: クラスタリング自体は24次元のXで行っており、PCAは結果を2次元(PC1, PC2)に射影して散布図に描くためだけに使う
X_pca = PCA(n_components=2).fit_transform(X)

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10")

for country, (x, y) in zip(oecd_bli_wide.index, X_pca):
    ax.annotate(country, xy=(x, y), fontsize=8,
                xytext=(3, 3), textcoords="offset points")

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(f"OECD Better Life Index - DBSCAN (eps={EPS}, min_samples={MIN_SAMPLES})")
ax.grid(True)
legend = ax.legend(*scatter.legend_elements(), title="Cluster (-1 = noise)")
ax.add_artist(legend)

outputs_dir = Path(__file__).resolve().parent / "outputs"
outputs_dir.mkdir(exist_ok=True)
fig.savefig(outputs_dir / "dbscan_clustering.png", dpi=300, bbox_inches="tight")
