# kMeansのクラスタ数kを1〜10で振り、inertia（エルボー法）とsilhouette_scoreを比較するスクリプト
#
# ノートブックには無い教師なし学習の体験用スクリプト
# 教師あり学習には無い「正解ラベルなしでの評価」という課題に触れる。
# エルボー法: inertia（クラスタ内二乗和）がk増加とともに単調減少する中で、
#             減少が緩やかになる「肘」の位置を目視で探す
# シルエットスコア: 各点が自分のクラスタにどれだけ馴染んでいるかを-1〜1で定量化する
#                    （kmeansと違い、直接kの最適値を数値で示せる）

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_produce.util.csv_io import read_csv
from data_produce.util.logging_config import setup_logger

logger = setup_logger(Path(__file__).stem)

oecd_bli_wide = read_csv("oecd_bli_wide.csv", index_col="Country")
X = StandardScaler().fit_transform(oecd_bli_wide.values)

# k=1ではsilhouette_scoreが定義できない（クラスタが1つしかないため）ので2から始める
k_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

best_k = list(k_range)[silhouette_scores.index(max(silhouette_scores))]
logger.info(f"silhouette_score best k: {best_k}")
for k, inertia, score in zip(k_range, inertias, silhouette_scores):
    logger.info(f"k={k}: inertia={inertia:.1f}, silhouette_score={score:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(list(k_range), inertias, "bo-")
ax1.set_xlabel("k")
ax1.set_ylabel("inertia")
ax1.set_title("Elbow method")
ax1.grid(True)

ax2.plot(list(k_range), silhouette_scores, "go-")
ax2.axvline(best_k, color="red", linestyle="--", label=f"best k={best_k}")
ax2.set_xlabel("k")
ax2.set_ylabel("silhouette score")
ax2.set_title("Silhouette score")
ax2.legend()
ax2.grid(True)

outputs_dir = Path(__file__).resolve().parent / "outputs"
outputs_dir.mkdir(exist_ok=True)
fig.savefig(outputs_dir / "elbow_silhouette.png", dpi=300, bbox_inches="tight")
