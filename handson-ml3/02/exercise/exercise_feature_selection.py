"""RandomForestRegressorベースのSelectFromModelを前処理後へ挿入し、
thresholdを変えたときの選択列数とSVRの交差検証RMSEを比較する（演習3）。

SVRのハイパーパラメータはexercise_svr.pyのRandomized Search（random_state=42固定）
で得られた最良パラメータ（kernel="rbf", C≈157055, gamma≈0.265）をそのまま使う。
ここでSVRの探索まで繰り返すと「特徴量選択の効果」と「SVRの調整」が混ざってしまうため、
SVR側は固定し、selector__thresholdだけを動かした差分に注目する。
計算量を抑えるため、exercise_svr.pyと同じく訓練データの先頭5,000件・3-foldを使う。
"""

from pathlib import Path
import sys

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
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
BEST_SVR_PARAMS = {
    "kernel": "rbf",
    "C": 157055.10989448498,
    "gamma": 0.26497040005002437,
}
THRESHOLDS: list[str | float] = ["median", "mean", 0.001, 0.005, 0.01, 0.02, 0.05]


def build_baseline_pipeline() -> Pipeline:
    """特徴量選択を挟まない、前処理→SVRのパイプライン（比較用の基準値）。"""

    return Pipeline([
        ("preprocessing", build_full_preprocessing()),
        ("svr", SVR(**BEST_SVR_PARAMS)),
    ])


def build_selector_pipeline(threshold: str | float) -> Pipeline:
    """前処理の後段にSelectFromModelを挟み、選択後の列だけをSVRへ渡すパイプライン。"""

    return Pipeline([
        ("preprocessing", build_full_preprocessing()),
        ("selector", SelectFromModel(
            RandomForestRegressor(random_state=RANDOM_STATE), threshold=threshold
        )),
        ("svr", SVR(**BEST_SVR_PARAMS)),
    ])


def evaluate_threshold(
    threshold: str | float, housing: pd.DataFrame, housing_labels: pd.Series
) -> tuple[float, list[int]]:
    """thresholdを固定したパイプラインをCVし、RMSEとfold毎の選択列数を返す。

    selectorはfold内の訓練データだけでfitされるため、fold間で選択列数が
    多少ばらつくことがある。return_estimator=Trueでfold毎のfit済みselectorを
    取り出し、リークなしで選択列数を数える。
    """

    pipeline = build_selector_pipeline(threshold)
    cv_result = cross_validate(
        pipeline, housing, housing_labels,
        scoring="neg_root_mean_squared_error", cv=CV_FOLDS,
        return_estimator=True,
    )
    rmse = -cv_result["test_score"].mean()
    n_selected = [
        int(estimator["selector"].get_support().sum())
        for estimator in cv_result["estimator"]
    ]
    return rmse, n_selected


def search_best_threshold(
    housing: pd.DataFrame, housing_labels: pd.Series
) -> GridSearchCV:
    """selector__thresholdをGridSearchCVの探索対象として扱い、最良値を探す。

    thresholdもハイパーパラメータの一つであり、他のパラメータと同じ探索機構に
    そのまま乗せられることを確認する。
    """

    pipeline = build_selector_pipeline(threshold="median")
    grid_search = GridSearchCV(
        pipeline, {"selector__threshold": THRESHOLDS},
        cv=CV_FOLDS, scoring="neg_root_mean_squared_error",
    )
    grid_search.fit(housing, housing_labels)
    return grid_search


def main() -> None:
    setup_logger(Path(__file__).stem)
    housing, housing_labels = load_features_and_labels()
    housing_subset = housing.iloc[:SUBSET_SIZE]
    labels_subset = housing_labels.iloc[:SUBSET_SIZE]
    logger.info(
        f"訓練セット全体: {housing.shape[0]:,}行 / "
        f"exercise_svr.pyと同じく先頭{SUBSET_SIZE:,}件・{CV_FOLDS}-foldに絞って比較する"
    )
    logger.info(f"固定するSVRパラメータ: {BEST_SVR_PARAMS}")

    logger.info("\n=== 基準値: 特徴量選択なし ===")
    baseline_rmses = cross_val_score(
        build_baseline_pipeline(), housing_subset, labels_subset,
        scoring="neg_root_mean_squared_error", cv=CV_FOLDS,
    )
    baseline_rmse = -baseline_rmses.mean()
    n_total_features = build_full_preprocessing().fit(
        housing_subset
    ).get_feature_names_out().shape[0]
    logger.info(f"前処理後の全列数: {n_total_features}")
    logger.info(f"RMSE（全列使用）: {baseline_rmse:,.0f}")

    logger.info(f"\n=== threshold別の選択列数とRMSE（{THRESHOLDS}） ===")
    rows = []
    for threshold in THRESHOLDS:
        rmse, n_selected = evaluate_threshold(threshold, housing_subset, labels_subset)
        rows.append({
            "threshold": threshold,
            "n_selected_min": min(n_selected),
            "n_selected_max": max(n_selected),
            "rmse": round(rmse),
            "rmse_diff_from_baseline": round(rmse - baseline_rmse),
        })
        logger.info(
            f"threshold={threshold!r}: 選択列数={n_selected}（全{n_total_features}列中）, "
            f"RMSE={rmse:,.0f}（基準値との差 {rmse - baseline_rmse:+,.0f}）"
        )
    result_df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    logger.info(f"\nRMSEが小さい順:\n{result_df.to_string(index=False)}")

    best_row = result_df.iloc[0]
    if best_row["rmse"] < baseline_rmse:
        logger.info(
            f"threshold={best_row['threshold']!r}は基準値より"
            f"{baseline_rmse - best_row['rmse']:,.0f}だけRMSEを改善した。"
        )
    else:
        logger.info(
            "どのthresholdも基準値（全列使用）のRMSEを下回らなかった。"
            "列を減らせば必ず性能が上がるわけではない。"
        )

    logger.info("\n=== GridSearchCVでselector__thresholdを探索 ===")
    grid_search = search_best_threshold(housing_subset, labels_subset)
    logger.info(f"最良threshold: {grid_search.best_params_['selector__threshold']!r}")
    logger.info(f"最良RMSE: {-grid_search.best_score_:,.0f}")
    logger.info(
        "selector__thresholdはPipeline内の他のパラメータと同じ"
        "'ステップ名__パラメータ名'の記法でGridSearchCV/RandomizedSearchCVに渡せる。"
        "つまりthresholdは「特徴量選択の副産物」ではなく、"
        "max_featuresやCと同列に扱えるハイパーパラメータの一つである。"
    )


if __name__ == "__main__":
    main()
