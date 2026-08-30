"""SVRを前処理パイプラインへ接続し、Grid Search / Randomized Searchで比較する（演習1・2）。

08_compare_models.pyで線形回帰・決定木・ランダムフォレストは比較済みのため、
ここではSVRを追加し、同じ条件（訓練データの先頭5,000件・3-fold）でランダムフォレストと
比較することに集中する。前処理はcommon.pyのbuild_full_preprocessing()を再利用する。
SVMは大規模データに弱いため、探索は先頭5,000件・3-foldへ絞って計算量を抑える。
"""

from pathlib import Path
import sys
import time

import pandas as pd
from loguru import logger
from scipy.stats import expon, loguniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

PROJECT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "learn"))
from common import build_full_preprocessing, load_features_and_labels  # noqa: E402
from data_produce.util.logging_config import setup_logger  # noqa: E402

CV_FOLDS = 3
RANDOM_STATE = 42
SUBSET_SIZE = 5000


def build_svr_pipeline() -> Pipeline:
    """前処理とSVRを1本のパイプラインへまとめる。"""

    return Pipeline([
        ("preprocessing", build_full_preprocessing()),
        ("svr", SVR()),
    ])


def run_grid_search(
    svr_pipeline: Pipeline,
    housing: pd.DataFrame,
    housing_labels: pd.Series,
    c_values_linear: list[float],
    c_values_rbf: list[float],
    gamma_values: list[float],
) -> tuple[GridSearchCV, float]:
    """線形カーネルのCと、RBFカーネルのC・gammaを総当たりで探索する。"""

    param_grid = [
        {"svr__kernel": ["linear"], "svr__C": c_values_linear},
        {"svr__kernel": ["rbf"], "svr__C": c_values_rbf, "svr__gamma": gamma_values},
    ]
    grid_search = GridSearchCV(
        svr_pipeline, param_grid, cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error",
    )
    start = time.perf_counter()
    grid_search.fit(housing, housing_labels)
    elapsed = time.perf_counter() - start
    return grid_search, elapsed


def run_randomized_search(
    svr_pipeline: Pipeline,
    housing: pd.DataFrame,
    housing_labels: pd.Series,
    n_iter: int = 50,
) -> tuple[RandomizedSearchCV, float]:
    """kernel・C（loguniform）・gamma（expon）を乱数探索する。"""

    param_distribs = {
        "svr__kernel": ["linear", "rbf"],
        "svr__C": loguniform(20, 200_000),
        "svr__gamma": expon(scale=1.0),
    }
    rnd_search = RandomizedSearchCV(
        svr_pipeline, param_distributions=param_distribs, n_iter=n_iter,
        cv=CV_FOLDS, scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
    )
    start = time.perf_counter()
    rnd_search.fit(housing, housing_labels)
    elapsed = time.perf_counter() - start
    return rnd_search, elapsed


def random_forest_cv_rmse(
    housing: pd.DataFrame, housing_labels: pd.Series, cv: int = CV_FOLDS
) -> float:
    """SVRと同じ行数・fold数でランダムフォレストのCV RMSEを計算する。"""

    rf_pipeline = Pipeline([
        ("preprocessing", build_full_preprocessing()),
        ("random_forest", RandomForestRegressor(random_state=RANDOM_STATE)),
    ])
    neg_rmses = cross_val_score(
        rf_pipeline, housing, housing_labels,
        scoring="neg_root_mean_squared_error", cv=cv,
    )
    return -neg_rmses.mean()


def main() -> None:
    setup_logger(Path(__file__).stem)
    housing, housing_labels = load_features_and_labels()
    housing_subset = housing.iloc[:SUBSET_SIZE]
    labels_subset = housing_labels.iloc[:SUBSET_SIZE]
    logger.info(
        f"訓練セット全体: {housing.shape[0]:,}行 / "
        f"SVMは大規模データに弱いため、探索には先頭{SUBSET_SIZE:,}件だけを使う"
    )

    svr_pipeline = build_svr_pipeline()

    logger.info(f"\n=== Grid Search 1回目（cv={CV_FOLDS}） ===")
    c_values_linear = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0]
    c_values_rbf = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    gamma_values = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    grid_search, grid_elapsed = run_grid_search(
        svr_pipeline, housing_subset, labels_subset,
        c_values_linear, c_values_rbf, gamma_values,
    )
    logger.info(f"実行時間: {grid_elapsed:.1f}秒")
    logger.info(f"最良パラメータ: {grid_search.best_params_}")
    logger.info(f"最良RMSE: {-grid_search.best_score_:,.0f}")

    best_kernel = grid_search.best_params_["svr__kernel"]
    best_c = grid_search.best_params_["svr__C"]
    c_candidates = c_values_linear if best_kernel == "linear" else c_values_rbf
    if best_c == max(c_candidates):
        logger.info(
            f"Cの最良値({best_c:,.0f})が探索範囲の上端に張り付いている。"
            "より大きいCも試す価値がある（探索範囲を上へ広げて2回目を実行する）。"
        )
        c_values_wide = [v * 10 for v in c_candidates]
        if best_kernel == "linear":
            grid_search_wide, grid_elapsed_wide = run_grid_search(
                svr_pipeline, housing_subset, labels_subset,
                c_values_wide, c_values_rbf, gamma_values,
            )
        else:
            grid_search_wide, grid_elapsed_wide = run_grid_search(
                svr_pipeline, housing_subset, labels_subset,
                c_values_linear, c_values_wide, gamma_values,
            )
        logger.info(f"\n=== Grid Search 2回目（探索範囲を拡張, cv={CV_FOLDS}） ===")
        logger.info(f"実行時間: {grid_elapsed_wide:.1f}秒")
        logger.info(f"最良パラメータ: {grid_search_wide.best_params_}")
        logger.info(f"最良RMSE: {-grid_search_wide.best_score_:,.0f}")
        if grid_search_wide.best_score_ > grid_search.best_score_:
            grid_search, grid_elapsed = grid_search_wide, grid_elapsed_wide

    logger.info(f"\n=== ランダムフォレストとの比較（同じ{SUBSET_SIZE:,}件・{CV_FOLDS}-fold） ===")
    rf_rmse = random_forest_cv_rmse(housing_subset, labels_subset)
    logger.info(f"SVR (Grid Search 最良) RMSE: {-grid_search.best_score_:,.0f}")
    logger.info(f"RandomForestRegressor RMSE: {rf_rmse:,.0f}")
    logger.info(
        "SVRはRandomForestRegressorより誤差が大きくなりやすい。"
        "ただし訓練データを5,000件に絞っているため、"
        "全訓練データを使った08_compare_models.pyの結果とは前提が異なる点に注意する。"
    )

    n_iter = 50
    logger.info(f"\n=== Randomized Search（n_iter={n_iter}, cv={CV_FOLDS}） ===")
    rnd_search, rnd_elapsed = run_randomized_search(
        svr_pipeline, housing_subset, labels_subset, n_iter=n_iter
    )
    logger.info(
        f"実行時間: {rnd_elapsed:.1f}秒（{n_iter}件 x {CV_FOLDS}-fold = {n_iter * CV_FOLDS}回学習）"
    )
    logger.info(f"最良パラメータ: {rnd_search.best_params_}")
    logger.info(f"最良RMSE: {-rnd_search.best_score_:,.0f}")

    logger.info("\n=== Grid Search と Randomized Search の比較（同じデータ・fold数） ===")
    logger.info(
        f"Grid Search: 実行時間={grid_elapsed:.1f}秒, 最良RMSE={-grid_search.best_score_:,.0f}"
    )
    logger.info(
        f"Randomized Search: 実行時間={rnd_elapsed:.1f}秒, 最良RMSE={-rnd_search.best_score_:,.0f}"
    )
    logger.info(
        "Cにloguniform、gammaにexponを使うことで、"
        "桁の異なる値も同じ試行回数のなかで効率よく探索できる。"
        "同程度の実行時間でどちらがより良いRMSEに到達するかを比較する材料になる。"
    )


if __name__ == "__main__":
    main()
