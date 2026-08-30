# oecd_bli.csv と gdp_per_capita.csv をダウンロードし、結合・整形して
# lifesat.csv（閾値内データ）と lifesat_full.csv（閾値外も含む完全データ）を data/ に生成するスクリプト
#
# https://github.com/ageron/data/blob/main/lifesat/gdp_per_capita.csv に対応するデータを手動で作成する方法

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from util.csv_io import data_path, exists, read_csv, write_csv
from util.logging_config import setup_logger

logger = setup_logger(Path(__file__).stem)

data_root = "https://github.com/ageron/data/raw/main/"
for filename in ("oecd_bli.csv", "gdp_per_capita.csv"):
    if not exists(filename):
        logger.info(f"Downloading {filename}")
        url = data_root + "lifesat/" + filename
        path = data_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, path)

oecd_bli = read_csv("oecd_bli.csv")
gdp_per_capita = read_csv("gdp_per_capita.csv")

gdp_year = 2020
gdppc_col = "GDP per capita (USD)"
lifesat_col = "Life satisfaction"

# 2020年のGDP per capitaのデータを抽出し、不要な列を削除して、列名を変更する
# 最終結果
# index: Country
# Columns: GDP per capita (USD)
gdp_per_capita = gdp_per_capita[gdp_per_capita["Year"] == gdp_year]
gdp_per_capita = gdp_per_capita.drop(["Code", "Year"], axis=1)
gdp_per_capita.columns = ["Country", gdppc_col]
gdp_per_capita.set_index("Country", inplace=True)

# OECDの生活満足度データを整形する
# 最終結果
# index: Country
# Columns: Life satisfaction を含む、様々な"indicator"のユニークな値
oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")

# Countryをインデックスにして、GDP per capitaと生活満足度のデータを結合する
full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                              left_index=True, right_index=True)
# GDP per capitaの列でソートする
full_country_stats.sort_values(by=gdppc_col, inplace=True)
# 不要な列を削除して、GDP per capitaと生活満足度の列だけを残す
full_country_stats = full_country_stats[[gdppc_col, lifesat_col]]


min_gdp = 23_500
max_gdp = 62_500

# GDP per capitaがmin_gdp以上max_gdp以下の国だけを抽出する
country_stats = full_country_stats[(full_country_stats[gdppc_col] >= min_gdp) &
                                   (full_country_stats[gdppc_col] <= max_gdp)]

write_csv(country_stats, "lifesat.csv")
write_csv(full_country_stats, "lifesat_full.csv")
