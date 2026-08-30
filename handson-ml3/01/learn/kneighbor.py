# k近傍法を用いて、GDP per capita から Life satisfaction の予測を行う写経スクリプト

import sys
from pathlib import Path

from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_produce.util.csv_io import read_csv
from data_produce.util.logging_config import setup_logger

logger = setup_logger(Path(__file__).stem)

# data_produce 層でダウンロード・作成済みのローカル CSV を読み込む
# https://github.com/ageron/data/blob/main/lifesat/lifesat.csv

# 3 columns:
# - Country
# - GDP per capita (USD)
# - Life satisfaction
lifesat = read_csv("lifesat.csv")

X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# Select a K-Nearest Neighbors model
model = KNeighborsRegressor(n_neighbors=3)

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[37_655.2]]  # Cyprus' GDP per capita in 2020
# 以下の3行が、予測に利用される近傍の国とそのGDPと満足度
# Slovenia,36547.7389559849,5.9
# Lithuania,36732.034744031,5.9
# Israel,38341.3075704083,7.2
logger.info(model.predict(X_new)) # outputss [[6.33333333]]


