# k分割交差検証（cross_val_score）で、1回のホールドアウトより安定した評価ができることを
# 確認する。foldごとのスコアのばらつき（標準偏差）も出力する
#
# train_test_evaluate.py・split_seed_sweep.pyで見た「1回の分割ではブレる」問題に対し、
# データをk個に分割してk回学習・評価を繰り返すことで、平均と標準偏差から
# 安定した汎化性能の見積もりを得る。

import sys
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_produce.util.csv_io import read_csv
from data_produce.util.logging_config import setup_logger

logger = setup_logger(Path(__file__).stem)

gdppc_col = "GDP per capita (USD)"
lifesat_col = "Life satisfaction"

lifesat = read_csv("lifesat.csv")
# [[gdppc_col]]と二重に囲むことで、Series（1次元）ではなくDataFrame（2次元）として取得する。
# fit(X, y)のXはshape (n_samples, n_features)の2次元配列である必要があるため
X = lifesat[[gdppc_col]].values
y = lifesat[lifesat_col].values

models = {
    "Linear": LinearRegression(),
    "KNN(k=3)": KNeighborsRegressor(n_neighbors=3),
}

k = 5  # 27件を5分割 -> 各foldのテストは約5〜6件

for name, model in models.items():
    r2_scores = cross_val_score(model, X, y, cv=k, scoring="r2")
    mse_scores = -cross_val_score(model, X, y, cv=k, scoring="neg_mean_squared_error")

    logger.info(f"{name:>10}: R^2 各fold  = {[f'{s:.3f}' for s in r2_scores]}")
    logger.info(f"{'':>10}  R^2 平均={r2_scores.mean():.3f} 標準偏差={r2_scores.std():.3f}")
    logger.info(f"{'':>10}  MSE 各fold  = {[f'{s:.3f}' for s in mse_scores]}")
    logger.info(f"{'':>10}  MSE 平均={mse_scores.mean():.3f} 標準偏差={mse_scores.std():.3f}")

# 【観察メモ】5回のfold結果を平均するため、split_seed_sweep.pyで見たような
# 「1回の分割だけに依存したブレ」は抑えられる。ただし標準偏差自体は、
# 小データ（各fold約5〜6件）ゆえに依然として小さくない点に注意。
