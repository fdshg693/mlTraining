# oecd_bli.csv（Country, Indicator, Value のlong format）を整形し、
# Countryを行、Indicatorを列に持つwideな oecd_bli_wide.csv を data/ に生成するスクリプト
#
# PCA/KMeans/DBSCANなど教師なし学習の学習用データとして使う（LEARNING_PLAN.md参照）

from pathlib import Path
from urllib.request import urlretrieve

from util.csv_io import data_path, exists, read_csv, write_csv
from util.logging_config import setup_logger

logger = setup_logger(Path(__file__).stem)

CSV_NAME = "oecd_bli.csv"

if not exists(CSV_NAME):
    logger.info(f"Downloading {CSV_NAME}")
    url = "https://github.com/ageron/data/raw/main/lifesat/" + CSV_NAME
    path = data_path(CSV_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, path)

oecd_bli = read_csv(CSV_NAME)

# 不平等区分(INEQUALITY)は"TOT"（全体）だけを使う
# 最終結果
# index: Country
# Columns: 24個のIndicatorのユニークな値
oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
oecd_bli_wide = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")

# "OECD - Total"はOECD全体の集計値であり個別の国ではないため、クラスタリング対象から除く
oecd_bli_wide = oecd_bli_wide.drop(index="OECD - Total")

# 欠損値の扱い: dropnaすると41ヶ国中25ヶ国が失われてしまう（欠損が0件の国は16ヶ国のみ）ため、
# 指標ごとの中央値で補完する。中央値は外れ値（例: Brazilの一部指標）の影響を受けにくい。
oecd_bli_wide = oecd_bli_wide.fillna(oecd_bli_wide.median())

write_csv(oecd_bli_wide, "oecd_bli_wide.csv")
