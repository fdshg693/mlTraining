# 線形回帰を用いて、GDP per capita から Life satisfaction の予測を行う写経スクリプト
# 閾値内のデータと閾値外のデータを含めた完全なデータセットで、線形回帰の結果を比較・描画する

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_produce.util.csv_io import read_csv
from data_produce.util.logging_config import setup_logger

logger = setup_logger(Path(__file__).stem)

# Download and prepare the data
# https://github.com/ageron/data/blob/main/lifesat/lifesat.csv

outputs_dir = Path(__file__).resolve().parent / "outputs"
outputs_dir.mkdir(exist_ok=True)

# 3 columns:
# - Country
# - GDP per capita (USD)
# - Life satisfaction

# 閾値内のデータ（data_produce 層で作成）
country_stats = read_csv("lifesat.csv", index_col="Country")
# 閾値外のデータも含めた完全なデータセット（data_produce 層で作成）
full_country_stats = read_csv("lifesat_full.csv", index_col="Country")

gdppc_col = "GDP per capita (USD)"
lifesat_col = "Life satisfaction"

min_gdp = 23_500
max_gdp = 62_500

min_life_sat = 4
max_life_sat = 9

X = country_stats[[gdppc_col]].values
y = country_stats[[lifesat_col]].values

# Select a linear model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction for Portugal
portugal_gdp_per_capita = country_stats[gdppc_col].loc["Portugal"] # Portugal's GDP per capita in 2020
logger.info(model.predict([[portugal_gdp_per_capita]])[0,0]) # outputss 5.930577613936961

t0, t1 = model.intercept_[0], model.coef_.ravel()[0]
logger.info(f"θ0={t0:.2f}, θ1={t1:.2e}")

full_country_stats.plot(kind='scatter', figsize=(8, 3),
                        x=gdppc_col, y=lifesat_col, grid=True)

missing_data = full_country_stats[(full_country_stats[gdppc_col] < min_gdp) |
                                  (full_country_stats[gdppc_col] > max_gdp)]

position_text_missing_countries = {
    "South Africa": (20_000, 4.2),
    "Colombia": (6_000, 8.2),
    "Brazil": (18_000, 7.8),
    "Mexico": (24_000, 7.4),
    "Chile": (30_000, 7.0),
    "Norway": (51_000, 6.2),
    "Switzerland": (62_000, 5.7),
    "Ireland": (81_000, 5.2),
    "Luxembourg": (92_000, 4.7),
}

for country, pos_text in position_text_missing_countries.items():
    pos_data_x, pos_data_y = missing_data.loc[country]
    plt.annotate(country, xy=(pos_data_x, pos_data_y),
                 xytext=pos_text, fontsize=12,
                 arrowprops=dict(facecolor='black', width=0.5,
                                 shrink=0.08, headwidth=5))
    plt.plot(pos_data_x, pos_data_y, "rs")

X = np.linspace(0, 115_000, 1000)
plt.plot(X, t0 + t1 * X, "b:")

lin_reg_full = LinearRegression()
Xfull = np.c_[full_country_stats[gdppc_col]]
yfull = np.c_[full_country_stats[lifesat_col]]
lin_reg_full.fit(Xfull, yfull)


t0full, t1full = lin_reg_full.intercept_[0], lin_reg_full.coef_.ravel()[0]
X = np.linspace(0, 115_000, 1000)
plt.plot(X, t0full + t1full * X, "k")

plt.axis([0, 115_000, min_life_sat, max_life_sat])

plt.savefig(outputs_dir / 'representative_training_data_scatterplot.png', dpi=300, bbox_inches="tight")