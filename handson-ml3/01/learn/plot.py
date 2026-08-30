# data_produce/data のCSVファイルを読み込み、GDP per capita と Life satisfaction の
# 散布図を描画する写経スクリプト

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_produce.util.csv_io import read_csv

# data_produce 層でダウンロード・作成済みのローカル CSV を読み込む
# https://github.com/ageron/data/blob/main/lifesat/lifesat.csv

# 3 columns:
# - Country
# - GDP per capita (USD)
# - Life satisfaction
lifesat = read_csv("lifesat.csv", index_col="Country")

X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# Visualize the data
lifesat.plot(kind='scatter', grid=True,
             x="GDP per capita (USD)", y="Life satisfaction")

position_text = {
    "Turkey": (29_500, 4.2),
    "Hungary": (28_000, 6.9),
    "France": (40_000, 5),
    "New Zealand": (28_000, 8.2),
    "Australia": (50_000, 5.5),
    "United States": (59_000, 5.3),
    "Denmark": (46_000, 8.5)
}

for country, pos_text in position_text.items():
    pos_data_x = lifesat["GDP per capita (USD)"].loc[country]
    pos_data_y = lifesat["Life satisfaction"].loc[country]
    country = "U.S." if country == "United States" else country
    plt.annotate(country, xy=(pos_data_x, pos_data_y),
                 xytext=pos_text, fontsize=12,
                 arrowprops=dict(facecolor='black', width=0.5,
                                 shrink=0.08, headwidth=5))
    plt.plot(pos_data_x, pos_data_y, "ro")


# 表示する範囲を指定する
# x: 23,500以上62,500以下
# y: 4以上9以下
plt.axis([23_500, 62_500, 4, 9])
outputs_dir = Path(__file__).resolve().parent / "outputs"
outputs_dir.mkdir(exist_ok=True)
plt.savefig(outputs_dir / "lifesat_scatter.png", dpi=300, bbox_inches="tight")