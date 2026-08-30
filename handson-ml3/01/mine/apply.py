# 任意の2列データに、線形回帰と KNN を当てて比較する汎用テンプレ
#
# 自分のデータでやる場合: data_produce/data/ に CSV を置き、下の3つを書き換えるだけ。
#   CSV_NAME    : 読み込む CSV のファイル名（data_produce/data 配下）
#   FEATURE_COL : 特徴量 x の列名
#   TARGET_COL  : 予測したい y の列名
#
# train_test_splitで訓練/テストを分離し、mean_squared_error・r2_scoreで
# 「訓練データへの当てはまり」だけでなく「未知データでの汎化性能」も数値で報告する

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

here = Path(__file__).resolve().parent
outputs_dir = here / "outputs"
outputs_dir.mkdir(exist_ok=True)

sys.path.insert(0, str(here.parent))
from data_produce.util.csv_io import read_csv
from data_produce.util.logging_config import setup_logger

logger = setup_logger(Path(__file__).stem)

# --- ここを自分のデータに合わせて変える -------------------------------
# デフォルトは data_produce 層の lifesat データ（そのまま実行できる）
CSV_NAME = "lifesat.csv"
FEATURE_COL = "GDP per capita (USD)"
TARGET_COL = "Life satisfaction"
# ----------------------------------------------------------------------

df = read_csv(CSV_NAME)
X = df[[FEATURE_COL]].values
y = df[TARGET_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear": LinearRegression(),
    "KNN(k=3)": KNeighborsRegressor(n_neighbors=3),
}

X_line = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)

plt.figure(figsize=(8, 4))
plt.scatter(X_train, y_train, color="black", label="train", zorder=3)
plt.scatter(X_test, y_test, color="red", marker="x", label="test", zorder=3)

for name, model in models.items():
    model.fit(X_train, y_train)

    # 訓練データへの当てはまり（見た目の良さ。汎化性能ではない）
    train_r2 = r2_score(y_train, model.predict(X_train))
    # 学習に使っていないテストデータでの性能（＝汎化性能）
    test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    logger.info(
        f"{name:>10}: train R^2={train_r2:.3f} | "
        f"test MSE={test_mse:.3f} test R^2={test_r2:.3f}"
    )
    plt.plot(X_line, model.predict(X_line), label=name)

plt.xlabel(FEATURE_COL)
plt.ylabel(TARGET_COL)
plt.legend()
plt.grid(True)
plt.savefig(outputs_dir / "apply.png", dpi=150, bbox_inches="tight")
logger.info(f"saved -> {outputs_dir / 'apply.png'}")
