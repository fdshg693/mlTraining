# KNN の k を変えて予測がどう変わるか観察する実験
#
# 【実行前に予想】k を 1 -> 20 と大きくすると、Cyprus の予測値と
# 予測曲線（青線）はどう変化する？ ガタガタになる？ なめらかになる？

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_produce.util.csv_io import read_csv
from data_produce.util.logging_config import setup_logger

logger = setup_logger(Path(__file__).stem)

outputs_dir = Path(__file__).resolve().parent / "outputs"
outputs_dir.mkdir(exist_ok=True)

gdppc_col = "GDP per capita (USD)"
lifesat_col = "Life satisfaction"

# data_produce 層で作った CSV を再利用する
lifesat = read_csv("lifesat.csv", index_col="Country")
X = lifesat[[gdppc_col]].values
y = lifesat[lifesat_col].values

cyprus_gdp = 37_655.2  # Cyprus の 2020 GDP per capita

# 予測曲線を描くための x 軸（min~max gdp の範囲）
X_line = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)

plt.figure(figsize=(8, 4))
plt.scatter(X, y, color="black", label="data", zorder=3)

for k in (1, 3, 7, 20):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X, y)
    pred = model.predict([[cyprus_gdp]])[0]
    logger.info(f"k={k:>2}: Cyprus 予測 = {pred:.3f}")
    plt.plot(X_line, model.predict(X_line), label=f"k={k}")

plt.axvline(cyprus_gdp, color="gray", linestyle=":", label="Cyprus")
plt.xlabel(gdppc_col)
plt.ylabel(lifesat_col)
plt.legend()
plt.grid(True)
plt.savefig(outputs_dir / "knn_k_sweep.png", dpi=150, bbox_inches="tight")
logger.info(f"saved -> {outputs_dir / 'knn_k_sweep.png'}")

# 【観察メモ】k=1 はデータ点を通る階段状（過学習）、k 大はなめらかで
# 全体平均に寄る（未学習側）。予想と合っていた？
