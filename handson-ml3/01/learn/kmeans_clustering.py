# 標準化した24指標（oecd_bli_wide.csv）にKMeansを適用し、国をk個のグループに分けるスクリプト
#
# ノートブックには無い教師なし学習（クラスタリング）の体験用スクリプト
# 「ラベルなしでグループを発見する」を体感する。PCAの2次元平面上にクラスタごとに色分けして可視化する

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_produce.util.csv_io import read_csv
from data_produce.util.logging_config import setup_logger

logger = setup_logger(Path(__file__).stem)

oecd_bli_wide = read_csv("oecd_bli_wide.csv", index_col="Country")

# クラスタリングもPCAと同じくスケールに敏感なため、指標ごとに標準化してから適用する
X = StandardScaler().fit_transform(oecd_bli_wide.values)

# k（クラスタ数）はここでは決め打ち。「正解ラベルなしでのkの選び方」は
# elbow_silhouette.py で扱う
N_CLUSTERS = 3
kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
labels = kmeans.fit_predict(X)

# 可視化用に2次元へ圧縮する（クラスタリング自体は24次元のXに対して行っている点に注意）
X_pca = PCA(n_components=2).fit_transform(X)

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10")

for country, (x, y) in zip(oecd_bli_wide.index, X_pca):
    ax.annotate(country, xy=(x, y), fontsize=8,
                xytext=(3, 3), textcoords="offset points")

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(f"OECD Better Life Index - KMeans clustering (k={N_CLUSTERS})")
ax.grid(True)
legend = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.add_artist(legend)

outputs_dir = Path(__file__).resolve().parent / "outputs"
outputs_dir.mkdir(exist_ok=True)
fig.savefig(outputs_dir / "kmeans_clustering.png", dpi=300, bbox_inches="tight")

for cluster_id in range(N_CLUSTERS):
    countries = oecd_bli_wide.index[labels == cluster_id].tolist()
    logger.info(f"cluster {cluster_id}: {countries}")
