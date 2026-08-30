# 標準化した24指標（oecd_bli_wide.csv）をPCAで2次元に圧縮し、国を散布図にプロットするスクリプト
#
# ノートブックには無い教師なし学習（次元削減）の体験用スクリプト
# 「高次元データでも意味のある構造が見える」を体感する

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_produce.util.csv_io import read_csv
from data_produce.util.logging_config import setup_logger

logger = setup_logger(Path(__file__).stem)

# data_produce 層で整形済みのローカル CSV を読み込む
# index: Country
# columns: 24個のIndicator（Life satisfaction, GDP per capita は含まない）
oecd_bli_wide = read_csv("oecd_bli_wide.csv", index_col="Country")

# PCAはスケールに敏感なため、指標ごとに平均0・分散1へ標準化してから適用する
X = StandardScaler().fit_transform(oecd_bli_wide.values)

# n_components: 元の24次元を何次元まで圧縮するか（＝抽出する主成分の数）
# 散布図としてそのまま目で見て確認したいので2を指定している
# （分散の説明率を優先するなら、explained_variance_ratio_の累積が
#   目標値（例: 0.8）を超える次元数を後から選ぶやり方もある）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 2つの主成分で元データの分散をどれだけ説明できているか
logger.info(f"explained_variance_ratio_: {pca.explained_variance_ratio_}")
logger.info(f"合計: {pca.explained_variance_ratio_.sum()}")

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(X_pca[:, 0], X_pca[:, 1])

for country, (x, y) in zip(oecd_bli_wide.index, X_pca):
    ax.annotate(country, xy=(x, y), fontsize=8,
                xytext=(3, 3), textcoords="offset points")

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("OECD Better Life Index (24 indicators) - PCA")
ax.grid(True)

outputs_dir = Path(__file__).resolve().parent / "outputs"
outputs_dir.mkdir(exist_ok=True)
fig.savefig(outputs_dir / "pca_visualize.png", dpi=300, bbox_inches="tight")
