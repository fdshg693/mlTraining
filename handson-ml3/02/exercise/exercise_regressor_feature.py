"""任意の回帰器の予測値を1列の特徴量として返すカスタム変換器FeatureFromRegressorを実装し、
緯度・経度からのKNN回帰予測をColumnTransformerの'geo'特徴量（クラスタ類似度）と
差し替えて性能を比較する（演習4・5）。

SVRのハイパーパラメータはexercise_svr.pyのRandomized Search（random_state=42固定）
で得られた最良パラメータをそのまま固定して使う。ここでSVRの探索まで同時に動かすと
「KNN予測特徴量とクラスタ類似度特徴量のどちらが有効か」という比較にSVR側のばらつきが
混ざってしまうため、まずSVRを固定した条件で比較し、最後にKNNの近傍数・重み・SVRの
パラメータを同時にRandomizedSearchCVで探索する。
計算量を抑えるため、exercise_svr.pyと同じく訓練データの先頭5,000件・3-foldを使う。
"""

from pathlib import Path
import sys
import time

import pandas as pd
from loguru import logger
from scipy.stats import expon, loguniform
from sklearn.base import BaseEstimator, MetaEstimatorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_array, check_is_fitted

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


class FeatureFromRegressor(MetaEstimatorMixin, TransformerMixin, BaseEstimator):
    """任意の回帰器をfitし、その予測値を1列の特徴量として返す変換器。

    KNeighborsRegressorに限定せず、MetaEstimatorMixinで`estimator`を
    必須パラメータとして持たせる。これによりget_params/set_paramsが
    内部の回帰器のハイパーパラメータも公開するため、
    GridSearchCV/RandomizedSearchCVの探索対象に'estimator__...'の形で
    そのまま乗せられる。
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        check_array(X)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_
        return self  # 必ずselfを返す

    def transform(self, X):
        check_is_fitted(self)
        predictions = self.estimator_.predict(X)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        return predictions

    def get_feature_names_out(self, names=None) -> list[str]:
        check_is_fitted(self)
        n_outputs = getattr(self.estimator_, "n_outputs_", 1)
        estimator_short_name = self.estimator_.__class__.__name__.lower()
        return [f"{estimator_short_name}_prediction_{i}" for i in range(n_outputs)]


def build_knn_geo_preprocessing(knn_transformer: FeatureFromRegressor) -> ColumnTransformer:
    """common.build_full_preprocessing()の'geo'ステップだけをKNN予測特徴量へ差し替える。"""

    base_preprocessing = build_full_preprocessing()
    transformers = [
        (name, knn_transformer if name == "geo" else clone(transformer), columns)
        for name, transformer, columns in base_preprocessing.transformers
    ]
    return ColumnTransformer(transformers, remainder=base_preprocessing.remainder)


def build_svr_pipeline(preprocessing: ColumnTransformer) -> Pipeline:
    """前処理とSVR（パラメータ固定）を1本のパイプラインへまとめる。"""

    return Pipeline([
        ("preprocessing", preprocessing),
        ("svr", SVR(**BEST_SVR_PARAMS)),
    ])


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

    logger.info("\n=== check_estimatorでscikit-learn APIへの適合を確認 ===")
    check_estimator(FeatureFromRegressor(KNeighborsRegressor()))
    logger.info("エラーなし。fit/transform/get_params/set_paramsの契約を満たしている。")

    logger.info("\n=== 緯度・経度からKNN回帰の予測値を1列の特徴量として作る ===")
    knn_reg = KNeighborsRegressor(n_neighbors=3, weights="distance")
    knn_transformer = FeatureFromRegressor(knn_reg)
    geo_features: pd.DataFrame = housing_subset[["latitude", "longitude"]]
    knn_predictions = knn_transformer.fit_transform(geo_features, labels_subset)
    logger.info(f"出力shape: {knn_predictions.shape}（入力{geo_features.shape[0]}行 x 1列）")
    logger.info(f"feature_names_in_: {list(knn_transformer.feature_names_in_)}")
    logger.info(f"get_feature_names_out(): {knn_transformer.get_feature_names_out()}")

    logger.info(f"\n=== クラスタ類似度 vs KNN予測特徴量（SVRパラメータ固定, cv={CV_FOLDS}） ===")
    cluster_pipeline = build_svr_pipeline(build_full_preprocessing())
    cluster_rmse = -cross_val_score(
        cluster_pipeline, housing_subset, labels_subset,
        scoring="neg_root_mean_squared_error", cv=CV_FOLDS,
    ).mean()
    logger.info(f"クラスタ類似度（cluster_simil, n_clusters=10）RMSE: {cluster_rmse:,.0f}")

    knn_geo_preprocessing = build_knn_geo_preprocessing(
        FeatureFromRegressor(KNeighborsRegressor(n_neighbors=3, weights="distance"))
    )
    knn_pipeline = build_svr_pipeline(knn_geo_preprocessing)
    knn_rmse = -cross_val_score(
        knn_pipeline, housing_subset, labels_subset,
        scoring="neg_root_mean_squared_error", cv=CV_FOLDS,
    ).mean()
    logger.info(f"KNN予測特徴量（n_neighbors=3, weights=distance）RMSE: {knn_rmse:,.0f}")
    if knn_rmse < cluster_rmse:
        logger.info(f"KNN予測特徴量の方が{cluster_rmse - knn_rmse:,.0f}だけRMSEが小さい。")
    else:
        logger.info(
            f"クラスタ類似度特徴量の方が{knn_rmse - cluster_rmse:,.0f}だけRMSEが小さい。"
            "KNNの近傍数・重みが未調整のままである点に注意する（次の探索で確認する）。"
        )

    logger.info("\n=== KNNの近傍数・重み・SVRパラメータを同時にRandomizedSearchCVで探索 ===")
    param_distribs = {
        "preprocessing__geo__estimator__n_neighbors": range(1, 30),
        "preprocessing__geo__estimator__weights": ["distance", "uniform"],
        "svr__C": loguniform(20, 200_000),
        "svr__gamma": expon(scale=1.0),
    }
    n_iter = 50
    rnd_search = RandomizedSearchCV(
        knn_pipeline, param_distributions=param_distribs, n_iter=n_iter,
        cv=CV_FOLDS, scoring="neg_root_mean_squared_error", random_state=RANDOM_STATE,
    )
    start = time.perf_counter()
    rnd_search.fit(housing_subset, labels_subset)
    elapsed = time.perf_counter() - start
    logger.info(
        f"実行時間: {elapsed:.1f}秒（{n_iter}件 x {CV_FOLDS}-fold = {n_iter * CV_FOLDS}回学習）"
    )
    logger.info(f"最良パラメータ: {rnd_search.best_params_}")
    logger.info(f"最良RMSE: {-rnd_search.best_score_:,.0f}")
    best_rmse = -rnd_search.best_score_
    logger.info(
        f"固定パラメータでのKNN特徴量RMSE({knn_rmse:,.0f})との比較: "
        f"{'改善した' if best_rmse < knn_rmse else '改善しなかった'}"
        f"（差 {knn_rmse - best_rmse:+,.0f}）。"
    )
    logger.info(
        f"クラスタ類似度特徴量のRMSE({cluster_rmse:,.0f})との比較: "
        f"{'上回った' if best_rmse < cluster_rmse else '下回らなかった'}"
        f"（差 {cluster_rmse - best_rmse:+,.0f}）。"
    )


if __name__ == "__main__":
    main()
