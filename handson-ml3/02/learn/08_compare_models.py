"""前処理込みのパイプラインで線形回帰・決定木・ランダムフォレストを比較する。

訓練誤差だけではモデルを選べないことを、交差検証誤差との差から確認する。
前処理（07_preprocessing_pipeline.py相当）はcommon.pyのbuild_full_preprocessing()
を再利用し、このファイルではモデル比較そのものに集中する。
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeRegressor

PROJECT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_DIR))
from common import build_full_preprocessing, load_features_and_labels
from data_produce.util.logging_config import setup_logger

CV_FOLDS = 10
RANDOM_STATE = 42


def build_models() -> dict[str, Pipeline]:
    """前処理と各回帰器を組み合わせたパイプラインを作る。"""

    return {
        "linear_regression": make_pipeline(
            build_full_preprocessing(), LinearRegression()
        ),
        "decision_tree": make_pipeline(
            build_full_preprocessing(),
            DecisionTreeRegressor(random_state=RANDOM_STATE),
        ),
        "random_forest": make_pipeline(
            build_full_preprocessing(),
            RandomForestRegressor(random_state=RANDOM_STATE),
        ),
    }


def training_rmse(model: Pipeline, housing: pd.DataFrame, housing_labels: pd.Series) -> float:
    """訓練セットへfitし、同じ訓練セットへの予測でRMSEを計算する。"""

    model.fit(housing, housing_labels)
    predictions = model.predict(housing)
    return root_mean_squared_error(housing_labels, predictions)


def cross_val_rmse(
    model: Pipeline, housing: pd.DataFrame, housing_labels: pd.Series, cv: int = CV_FOLDS
) -> pd.Series:
    """cv分割の交差検証RMSEをSeriesとして返す（scoringの符号を反転させる）。"""

    neg_rmses = cross_val_score(
        model, housing, housing_labels,
        scoring="neg_root_mean_squared_error", cv=cv,
    )
    return pd.Series(-neg_rmses)


def main() -> None:
    setup_logger(Path(__file__).stem)
    housing, housing_labels = load_features_and_labels()
    logger.info(f"訓練セット（特徴量）: {housing.shape[0]:,}行 x {housing.shape[1]}列")

    models = build_models()

    logger.info("\n=== 線形回帰: 訓練セットでの予測例 ===")
    lin_reg = models["linear_regression"]
    lin_reg.fit(housing, housing_labels)
    sample_predictions: np.ndarray = lin_reg.predict(housing)[:5].round(-2)
    sample_labels: np.ndarray = housing_labels.iloc[:5].to_numpy()
    # 両方ともNumPy配列なので、/と-1は要素ごと（element-wise）に適用される
    error_ratios = sample_predictions / sample_labels - 1
    logger.info(f"予測値（100ドル単位に丸め）: {sample_predictions}")
    logger.info(f"実際の値: {sample_labels}")
    logger.info(
        "誤差率: " + ", ".join(f"{100 * ratio:.1f}%" for ratio in error_ratios)
    )

    logger.info("\n=== 訓練誤差（RMSE） ===")
    train_rmses = {}
    for name, model in models.items():
        train_rmses[name] = training_rmse(model, housing, housing_labels)
        logger.info(f"{name}: 訓練RMSE = {train_rmses[name]:,.0f}")
    logger.info(
        "decision_treeの訓練RMSEが極端に小さい場合、訓練データへの過剰適合を疑う必要がある"
        "（木がリーフ1つに1サンプルまで分割し、訓練データを丸暗記している可能性がある）。"
    )

    logger.info(f"\n=== 交差検証誤差（RMSE、{CV_FOLDS}-fold） ===")
    cv_rmses = {}
    for name, model in models.items():
        cv_rmses[name] = cross_val_rmse(model, housing, housing_labels)
        stats = cv_rmses[name].describe()
        logger.info(
            f"{name}: mean={stats['mean']:,.0f}  std={stats['std']:,.0f}  "
            f"min={stats['min']:,.0f}  max={stats['max']:,.0f}"
        )

    logger.info("\n=== 訓練誤差 vs 交差検証誤差 ===")
    for name in models:
        gap = cv_rmses[name].mean() - train_rmses[name]
        logger.info(
            f"{name}: 訓練RMSE={train_rmses[name]:,.0f}  "
            f"交差検証RMSE(平均)={cv_rmses[name].mean():,.0f}  差={gap:,.0f}"
        )
    logger.info(
        "decision_treeは訓練RMSEがほぼ0でも交差検証RMSEはrandom_forestより悪化しやすく、"
        "訓練データへの過学習が未知データへの性能に表れている。"
        "random_forestは複数の木の平均を取ることで、単一の決定木より過学習を抑えられる。"
        "linear_regressionは訓練誤差と交差検証誤差の差が小さいが、"
        "そもそもの誤差水準は決定木・ランダムフォレストより大きい（過小適合の傾向）。"
    )


if __name__ == "__main__":
    main()
