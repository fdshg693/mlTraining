# 10次多項式を含めて過学習を示すためのモデルを作成する写経スクリプト

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, pipeline, preprocessing

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_produce.util.csv_io import read_csv

outputs_dir = Path(__file__).resolve().parent / "outputs"
outputs_dir.mkdir(exist_ok=True)

gdppc_col = "GDP per capita (USD)"
lifesat_col = "Life satisfaction"

min_life_sat = 4
max_life_sat = 9

# Threshold-excluded countries are included in this full dataset (data_produce 層で作成).
full_country_stats = read_csv("lifesat_full.csv", index_col="Country")

Xfull = full_country_stats[[gdppc_col]].values
yfull = full_country_stats[[lifesat_col]].values

full_country_stats.plot(
    kind="scatter",
    figsize=(8, 3),
    x=gdppc_col,
    y=lifesat_col,
    grid=True,
)

# 10次多項式を含めて過学習を示すためのモデルを作成する
poly = preprocessing.PolynomialFeatures(degree=10, include_bias=False)
scaler = preprocessing.StandardScaler()
lin_reg2 = linear_model.LinearRegression()

pipeline_reg = pipeline.Pipeline([
    ("poly", poly),
    ("scal", scaler),
    ("lin", lin_reg2),
])

pipeline_reg.fit(Xfull, yfull)

X = np.linspace(0, 115_000, 1000)
curve = pipeline_reg.predict(X[:, np.newaxis])
plt.plot(X, curve)

plt.axis([0, 115_000, min_life_sat, max_life_sat])

plt.savefig(outputs_dir / "overfitting_model_plot.png", dpi=300, bbox_inches="tight")
