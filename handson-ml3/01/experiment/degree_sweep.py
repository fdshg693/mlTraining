# 多項式回帰の degree を変えて、過学習が育つ様子を観察する実験
#
# 【実行前に予想】degree を 1 -> 30 と上げると曲線はどうなる？
# データ点の近くは？ データのない端（外挿）は？

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, pipeline, preprocessing

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_produce.util.csv_io import read_csv
from data_produce.util.logging_config import setup_logger

logger = setup_logger(Path(__file__).stem)

outputs_dir = Path(__file__).resolve().parent / "outputs"
outputs_dir.mkdir(exist_ok=True)

gdppc_col = "GDP per capita (USD)"
lifesat_col = "Life satisfaction"

# 閾値外も含む完全データ（data_produce 層で作成）
full = read_csv("lifesat_full.csv", index_col="Country")
X = full[[gdppc_col]].values
y = full[lifesat_col].values

X_line = np.linspace(0, 115_000, 1000).reshape(-1, 1)

plt.figure(figsize=(8, 4))
plt.scatter(X, y, color="black", label="data", zorder=3)

for degree in (1, 2, 3, 10, 30):
    reg = pipeline.Pipeline([
        ("poly", preprocessing.PolynomialFeatures(degree=degree, include_bias=False)),
        ("scal", preprocessing.StandardScaler()),
        ("lin", linear_model.LinearRegression()),
    ])
    reg.fit(X, y)
    plt.plot(X_line, reg.predict(X_line), label=f"degree={degree}")

plt.axis([0, 115_000, 4, 9])  # y 軸を生活満足度の現実的な範囲に固定
plt.xlabel(gdppc_col)
plt.ylabel(lifesat_col)
plt.legend()
plt.grid(True)
plt.savefig(outputs_dir / "degree_sweep.png", dpi=150, bbox_inches="tight")
logger.info(f"saved -> {outputs_dir / 'degree_sweep.png'}")