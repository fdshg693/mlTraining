"""複数のlearn/スクリプトで再利用する前処理パイプラインとデータ読込。

06_custom_transformers.pyと07_preprocessing_pipeline.pyで組み立てた
前処理は、08以降のモデル比較・チューニング・最終評価でもそのまま使う。
ここでは前処理をどう作るかではなく、完成した前処理を再現よく使い回すことに
役割を絞る。カスタム変換器の実装そのものを学ぶ場合は06を参照する。
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

PROJECT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_DIR))
from data_produce.util.file_io import data_path, read_csv  # noqa: E402

HOUSING_TRAIN_CSV = "housing_train.csv"
HOUSING_TEST_CSV = "housing_test.csv"
TARGET_COLUMN = "median_house_value"
NUM_ATTRIBS = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
]
CAT_ATTRIBS = ["ocean_proximity"]


def load_features_and_labels(
    csv_name: str = HOUSING_TRAIN_CSV,
) -> tuple[pd.DataFrame, pd.Series]:
    """02_split_data.pyが保存したCSVを読み込み、特徴量とラベルに分ける。"""

    if not data_path(csv_name).is_file():
        raise FileNotFoundError(
            f"{csv_name}が見つかりません。先にlearn/02_split_data.pyを実行してください。"
        )
    housing = read_csv(csv_name)
    housing_labels = housing[TARGET_COLUMN].copy()
    housing_features = housing.drop(columns=[TARGET_COLUMN])
    return housing_features, housing_labels


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """KMeansのクラスタ中心とのRBF類似度を特徴量として出力する変換器。

    実装の詳細（fit/transform契約、check_is_fittedの役割）は
    06_custom_transformers.pyで確認済みのものと同じ。
    """

    def __init__(self, n_clusters: int = 10, gamma: float = 1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            self.n_clusters, n_init=10, random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        check_is_fitted(self)
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None) -> list[str]:
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


def _column_ratio(X: np.ndarray) -> np.ndarray:
    return X[:, [0]] / X[:, [1]]


def _ratio_name(function_transformer: FunctionTransformer, feature_names_in) -> list[str]:
    return ["ratio"]


def ratio_pipeline() -> Pipeline:
    """欠損補完→列比率→標準化の3ステップパイプライン。"""

    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(_column_ratio, feature_names_out=_ratio_name),
        StandardScaler(),
    )


def build_num_pipeline() -> Pipeline:
    """数値列へ「中央値補完→標準化」を適用するパイプライン。"""

    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


def build_log_pipeline() -> Pipeline:
    """欠損補完→対数変換→標準化の3ステップパイプライン。"""

    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler(),
    )


def build_cat_pipeline() -> Pipeline:
    """カテゴリ列へ「最頻値補完→未知値を無視するOne-Hot」を適用するパイプライン。"""

    return make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )


def build_full_preprocessing() -> ColumnTransformer:
    """比率・対数・クラスタ類似度・カテゴリ変換を1つのColumnTransformerへ統合する。

    remainderへ渡した数値パイプラインが、他のどの変換にも指定されなかった
    housing_median_ageを「中央値補完→標準化」で処理する。
    """

    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)

    return ColumnTransformer(
        [
            ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
            ("people_per_house", ratio_pipeline(), ["population", "households"]),
            (
                "log",
                build_log_pipeline(),
                ["total_bedrooms", "total_rooms", "population", "households",
                 "median_income"],
            ),
            ("geo", cluster_simil, ["latitude", "longitude"]),
            ("cat", build_cat_pipeline(), make_column_selector(dtype_include=object)),
        ],
        remainder=build_num_pipeline(),
    )
