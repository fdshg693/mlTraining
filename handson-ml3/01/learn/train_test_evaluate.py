# ホールドアウト検証: train_test_splitで訓練/テストを分離し、
# 「全データで学習した場合の見た目の良さ」と「未知データでの実際の性能」の差を確認する
#
# ノートブックには無い評価手法の学習用スクリプト

import sys
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_produce.util.csv_io import read_csv
from data_produce.util.logging_config import setup_logger

logger = setup_logger(Path(__file__).stem)

gdppc_col = "GDP per capita (USD)"
lifesat_col = "Life satisfaction"

lifesat = read_csv("lifesat.csv")
X = lifesat[[gdppc_col]].values
y = lifesat[lifesat_col].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
logger.info(f"train={len(X_train)}件, test={len(X_test)}件")

models = {
    "Linear": LinearRegression(),
    "KNN(k=3)": KNeighborsRegressor(n_neighbors=3),
}

for name, model in models.items():
    model.fit(X_train, y_train)

    # 訓練データそのものへの当てはまり（見た目の良さ。汎化性能ではない）
    train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)

    # 学習に使っていないテストデータでの性能（＝汎化性能）
    test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    logger.info(
        f"{name:>10}: train MSE={train_mse:.3f} R^2={train_r2:.3f} | "
        f"test MSE={test_mse:.3f} R^2={test_r2:.3f}"
    )

# 【観察メモ】train側の指標はどちらのモデルも良く見えがちだが、
# test側（未知データ）で初めて実際の汎化性能が分かる。
# lifesatは27件しかないため、test側の指標はsplit_seed_sweep.pyで見る通り
# 分割の仕方（乱数シード）次第で大きくブレる点に注意。
