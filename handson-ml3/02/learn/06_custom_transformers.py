"""scikit-learnのAPI契約に沿ったカスタム変換器を作る。

FunctionTransformerで関数をそのまま変換器化する方法と、
BaseEstimator/TransformerMixinを継承して学習済み属性を持つ変換器を
自作する方法の両方を確認する。
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

PROJECT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_DIR))
from data_produce.util.file_io import data_path, read_csv, save_figure
from data_produce.util.logging_config import setup_logger

HOUSING_TRAIN_CSV = "housing_train.csv"
TARGET_COLUMN = "median_house_value"
SF_COORDS = (37.7749, -122.41)


def load_features_and_labels() -> tuple[pd.DataFrame, pd.Series]:
    """02_split_data.pyが保存した訓練セットを読み込み、特徴量とラベルに分ける。"""

    if not data_path(HOUSING_TRAIN_CSV).is_file():
        raise FileNotFoundError(
            f"{HOUSING_TRAIN_CSV}が見つかりません。"
            "先にlearn/02_split_data.pyを実行してください。"
        )
    housing = read_csv(HOUSING_TRAIN_CSV)
    housing_labels = housing[TARGET_COLUMN].copy()
    housing_features = housing.drop(columns=[TARGET_COLUMN])
    return housing_features, housing_labels


def build_function_transformers() -> dict[str, FunctionTransformer]:
    """関数をそのまま変換器として使うFunctionTransformerの例を集める。"""

    return {
        "log": FunctionTransformer(np.log, inverse_func=np.exp),
        "age_similarity": FunctionTransformer(
            rbf_kernel, kw_args=dict(Y=[[35.0]], gamma=0.1)
        ),
        "sf_similarity": FunctionTransformer(
            rbf_kernel, kw_args=dict(Y=[SF_COORDS], gamma=0.1)
        ),
        "ratio": FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]]),
    }


def demo_function_transformers(
    housing: pd.DataFrame, transformers: dict[str, FunctionTransformer]
) -> dict[str, np.ndarray]:
    """各FunctionTransformerを実際の列へ適用し、先頭の出力を返す。"""

    log_population = transformers["log"].transform(housing[["population"]])
    age_similarity = transformers["age_similarity"].transform(
        housing[["housing_median_age"]]
    )
    sf_similarity = transformers["sf_similarity"].transform(
        housing[["latitude", "longitude"]]
    )
    bedrooms_ratio = transformers["ratio"].transform(
        housing[["total_bedrooms", "total_rooms"]].to_numpy()
    )

    return {
        "log_population": log_population.to_numpy().ravel(),
        "age_similarity": age_similarity.ravel(),
        "sf_similarity": sf_similarity.ravel(),
        "bedrooms_ratio": bedrooms_ratio.ravel(),
    }


class StandardScalerClone(BaseEstimator, TransformerMixin):
    """StandardScalerの契約を確認するための簡易再実装。

    __init__には学習対象でないハイパーパラメータ（with_mean）だけを置き、
    fit()で学習した属性はmean_・scale_・n_features_in_のように
    末尾アンダースコア付きで保持する。
    """

    def __init__(self, with_mean: bool = True):
        self.with_mean = with_mean

    def fit(self, X, y=None):
        X = check_array(X)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"入力の列数{X.shape[1]}がfit時の列数{self.n_features_in_}と一致しません。"
            )
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_


def compare_scaler_clone(housing_num: pd.DataFrame) -> dict[str, np.ndarray]:
    """StandardScalerCloneとsklearn.StandardScalerの出力を比較する。

    StandardScalerClone・StandardScalerともに欠損値を扱えないため、
    04_clean_encode.pyと同様にtotal_bedroomsを中央値で補完してから比較する。
    """

    housing_num_imputed = SimpleImputer(strategy="median").fit_transform(housing_num)

    clone_scaler = StandardScalerClone()
    clone_scaled = clone_scaler.fit_transform(housing_num_imputed)

    sklearn_scaler = StandardScaler()
    sklearn_scaled = sklearn_scaler.fit_transform(housing_num_imputed)

    return {"clone": clone_scaled, "sklearn": sklearn_scaled}


def transform_before_fit_raises() -> bool:
    """fit前にtransform()を呼ぶとNotFittedErrorになることを確認する。"""

    unfitted_scaler = StandardScalerClone()
    try:
        unfitted_scaler.transform(np.array([[1.0, 2.0]]))
    except NotFittedError:
        return True
    return False


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """KMeansのクラスタ中心とのRBF類似度を特徴量として出力する変換器。"""

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


def cluster_similarity_features(
    housing: pd.DataFrame,
    n_clusters: int = 10,
    gamma: float = 1.0,
    random_state: int = 42,
) -> tuple[np.ndarray, ClusterSimilarity]:
    """緯度・経度からクラスタ類似度特徴量を作る。"""

    cluster_simil = ClusterSimilarity(
        n_clusters=n_clusters, gamma=gamma, random_state=random_state
    )
    similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]])
    return similarities, cluster_simil


def plot_cluster_similarity(
    housing: pd.DataFrame,
    similarities: np.ndarray,
    cluster_simil: ClusterSimilarity,
    outputs_dir: str | Path | None = None,
) -> Path:
    """地区ごとの最大クラスタ類似度と、クラスタ中心の位置を地図上に描く。"""

    housing_renamed = housing.rename(
        columns={
            "latitude": "Latitude",
            "longitude": "Longitude",
            "population": "Population",
        }
    )
    # similarities は (地区数, n_clusters) の行列。axis=1 で列方向に集約し、
    # 地区ごとに最も近いクラスタへの類似度（スカラー）を求める。
    housing_renamed["Max cluster similarity"] = similarities.max(axis=1)

    fig, ax = plt.subplots(figsize=(10, 7))
    housing_renamed.plot(
        kind="scatter",
        x="Longitude",
        y="Latitude",
        grid=True,
        s=housing_renamed["Population"] / 100,
        label="Population",
        c="Max cluster similarity",
        cmap="jet",
        colorbar=True,
        legend=True,
        sharex=False,
        ax=ax,
    )
    ax.plot(
        cluster_simil.kmeans_.cluster_centers_[:, 1],
        cluster_simil.kmeans_.cluster_centers_[:, 0],
        linestyle="",
        color="black",
        marker="X",
        markersize=20,
        label="Cluster centers",
    )
    ax.legend(loc="upper right")
    fig.tight_layout()

    if outputs_dir is None:
        path = save_figure(fig, "district_cluster_plot.png", dpi=300)
    else:
        path = save_figure(fig, "district_cluster_plot.png", outputs_dir, dpi=300)
    plt.close(fig)
    return path


def main() -> None:
    setup_logger(Path(__file__).stem)
    housing, _housing_labels = load_features_and_labels()
    housing_num = housing.select_dtypes(include=[np.number])
    logger.info(f"訓練セット（数値列）: {housing_num.shape[0]:,}行 x {housing_num.shape[1]}列")

    logger.info("\n=== FunctionTransformerで関数をそのまま変換器化する ===")
    transformers = build_function_transformers()
    demo_outputs = demo_function_transformers(housing, transformers)
    logger.info(f"log(population) 先頭5件: {demo_outputs['log_population'][:5].round(3)}")
    logger.info(f"age_similarity(35歳基準) 先頭5件: {demo_outputs['age_similarity'][:5].round(3)}")
    logger.info(f"sf_similarity(サンフランシスコ基準) 先頭5件: {demo_outputs['sf_similarity'][:5].round(4)}")
    logger.info(f"bedrooms_ratio 先頭5件: {demo_outputs['bedrooms_ratio'][:5].round(3)}")
    logger.info(
        "FunctionTransformerはfitで学習する状態を持たないため、"
        "対数変換・RBF類似度・列比率のような無状態な処理をパイプラインへ"
        "組み込む場合に手早く使える。"
    )

    logger.info("\n=== StandardScalerCloneとStandardScalerの比較 ===")
    scaled = compare_scaler_clone(housing_num)
    all_close = np.allclose(scaled["clone"], scaled["sklearn"])
    logger.info(f"両者の出力が一致するか: {all_close}")
    fit_before_transform_ok = transform_before_fit_raises()
    logger.info(
        f"fit前にtransform()を呼ぶとNotFittedErrorになるか: {fit_before_transform_ok}"
    )
    logger.info(
        "__init__には学習対象でないwith_meanだけを保存し、fit()で計算した"
        "mean_・scale_・n_features_in_をtransform()側で使う。"
        "check_is_fittedにより、fit前のtransform()呼び出しを検出できる。"
    )

    logger.info("\n=== ClusterSimilarityによるクラスタ類似度特徴量 ===")
    similarities, cluster_simil = cluster_similarity_features(housing)
    logger.info(f"類似度の出力shape: {similarities.shape}")
    logger.info(f"先頭3件（丸め）:\n{np.round(similarities[:3], 2)}")
    logger.info(f"get_feature_names_out(): {cluster_simil.get_feature_names_out()[:3]}...")
    cluster_plot_path = plot_cluster_similarity(housing, similarities, cluster_simil)
    logger.info(f"保存しました: {cluster_plot_path}")
    logger.info(
        "ClusterSimilarityはKMeansのクラスタ番号をカテゴリとして使うのではなく、"
        "各クラスタ中心との連続的なRBF類似度を特徴量にする。"
        "get_feature_names_out()を実装しておくことで、Pipeline/ColumnTransformerと"
        "組み合わせた後も出力列の意味を追跡できる。"
    )


if __name__ == "__main__":
    main()
