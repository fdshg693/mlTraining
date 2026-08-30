"""訓練セットの数値列に対するスケーリングと分布変換を確認する。

target(median_house_value)は特徴量のスケーリング対象からは外すが、
末尾のTransformedTargetRegressorの節では目的変数側の変換を扱う。
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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


def _save(figure, file_name: str, outputs_dir: str | Path | None) -> Path:
    if outputs_dir is None:
        return save_figure(figure, file_name, dpi=300)
    return save_figure(figure, file_name, outputs_dir, dpi=300)


def compare_scalers(
    housing_num: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler, StandardScaler]:
    """MinMaxScaler(-1, 1)とStandardScalerの出力を同じ列でDataFrame化して比較する。"""

    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    min_max_array = min_max_scaler.fit_transform(housing_num)
    min_max_df = pd.DataFrame(
        min_max_array, columns=housing_num.columns, index=housing_num.index
    )

    std_scaler = StandardScaler()
    std_array = std_scaler.fit_transform(housing_num)
    std_df = pd.DataFrame(std_array, columns=housing_num.columns, index=housing_num.index)

    return min_max_df, std_df, min_max_scaler, std_scaler


def plot_log_population(
    population: pd.Series,
    outputs_dir: str | Path | None = None,
) -> Path:
    """populationの元の分布と対数変換後の分布を並べて保存する。"""

    fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    population.hist(ax=axs[0], bins=50)
    population.apply(np.log).hist(ax=axs[1], bins=50)
    axs[0].set_xlabel("population")
    axs[1].set_xlabel("log(population)")
    axs[0].set_ylabel("districts")
    fig.tight_layout()
    path = _save(fig, "long_tail_plot.png", outputs_dir)
    plt.close(fig)
    return path


def percentile_transform(income: pd.Series) -> pd.Series:
    """median_incomeを1〜100のパーセンタイル順位へ変換する。"""

    percentiles = [np.percentile(income, p) for p in range(1, 100)]
    return pd.cut(
        income,
        bins=[-np.inf, *percentiles, np.inf],
        labels=range(1, 101),
    )


def plot_percentile_income(
    flattened_income: pd.Series,
    outputs_dir: str | Path | None = None,
) -> Path:
    """パーセンタイル変換後のmedian_incomeがほぼ一様分布になることを図で確認する。"""

    fig, ax = plt.subplots()
    flattened_income.astype(int).hist(bins=50, ax=ax)
    ax.set_xlabel("median_income percentile")
    ax.set_ylabel("districts")
    fig.tight_layout()
    path = _save(fig, "income_percentile_hist.png", outputs_dir)
    plt.close(fig)
    return path


def rbf_age_similarity(
    age: pd.Series,
    target_age: float = 35.0,
    gammas: tuple[float, ...] = (0.1, 0.03),
) -> dict[float, np.ndarray]:
    """housing_median_ageと基準年齢とのRBF類似度をgammaごとに計算する。"""

    # RBF類似度 = exp(-gamma * (age - target_age)^2)。
    # target_ageに近いほど1に近づき、離れるほど0に近づくガウス型の類似度で、
    # gammaが大きいほど減衰が急峻(=似ているとみなす範囲が狭い)になる。
    return {
        gamma: rbf_kernel(age.to_frame(), [[target_age]], gamma=gamma).ravel()
        for gamma in gammas
    }


def plot_age_similarity(
    age: pd.Series,
    target_age: float = 35.0,
    gammas: tuple[float, ...] = (0.1, 0.03),
    outputs_dir: str | Path | None = None,
) -> Path:
    """年齢のヒストグラムと、gamma別の年齢類似度カーブを重ねて保存する。"""

    ages_range = np.linspace(age.min(), age.max(), 500).reshape(-1, 1)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("housing_median_age")
    ax1.set_ylabel("districts")
    ax1.hist(age, bins=50)

    # 同じX軸を共有しながら、異なる位置のY軸を持つグラフを重ねる。
    # Y軸がX軸の右の終端に出来る
    ax2 = ax1.twinx()
    for gamma in gammas:
        similarity = rbf_kernel(ages_range, [[target_age]], gamma=gamma)
        ax2.plot(ages_range, similarity, label=f"gamma = {gamma}")
    ax2.set_ylabel("age similarity")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    path = _save(fig, "age_similarity_plot.png", outputs_dir)
    plt.close(fig)
    return path


def rbf_geo_similarity_sf(
    housing: pd.DataFrame,
    sf_coords: tuple[float, float] = SF_COORDS,
    gamma: float = 0.1,
) -> np.ndarray:
    """緯度・経度とサンフランシスコ座標とのRBF類似度を計算する。"""

    return rbf_kernel(housing[["latitude", "longitude"]], [sf_coords], gamma=gamma).ravel()


def compare_target_scaling(
    housing: pd.DataFrame,
    housing_labels: pd.Series,
) -> dict[str, np.ndarray]:
    """目的変数の手動スケーリングとTransformedTargetRegressorの予測が一致するか確認する。"""

    some_new_data = housing[["median_income"]].iloc[:5]

    target_scaler = StandardScaler()
    scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())
    manual_model = LinearRegression()
    manual_model.fit(housing[["median_income"]], scaled_labels)
    manual_predictions = target_scaler.inverse_transform(
        manual_model.predict(some_new_data)
    ).ravel()

    wrapped_model = TransformedTargetRegressor(
        LinearRegression(), transformer=StandardScaler()
    )
    wrapped_model.fit(housing[["median_income"]], housing_labels)
    wrapped_predictions = wrapped_model.predict(some_new_data)

    return {
        "manual": manual_predictions,
        "wrapped": wrapped_predictions,
    }


def main() -> None:
    setup_logger(Path(__file__).stem)
    housing, housing_labels = load_features_and_labels()
    housing_num = housing.select_dtypes(include=[np.number])
    logger.info(f"訓練セット（数値列）: {housing_num.shape[0]:,}行 x {housing_num.shape[1]}列")

    logger.info("\n=== MinMaxScaler(-1, 1) と StandardScaler の比較 ===")
    min_max_df, std_df, min_max_scaler, std_scaler = compare_scalers(housing_num)
    logger.info(f"MinMaxScaler後のmin/max（先頭3列）:\n{min_max_df.iloc[:, :3].agg(['min', 'max'])}")
    logger.info(f"StandardScaler後のmean/std（先頭3列、丸め）:\n"
                f"{std_df.iloc[:, :3].agg(['mean', 'std']).round(3)}")
    logger.info(
        "MinMaxScalerは指定した範囲へ値を引き伸ばすため外れ値の影響を受けやすく、"
        "StandardScalerは平均0・分散1へそろえるため外れ値があっても範囲が固定されない。"
    )

    logger.info("\n=== populationの対数変換 ===")
    long_tail_path = plot_log_population(housing["population"])
    logger.info(f"保存しました: {long_tail_path}")
    logger.info(
        "populationは右に長い裾を持つが、対数変換後はより左右対称に近づく。"
        "長い裾を持つ特徴量はそのままだと大きな値に予測が引っ張られやすい。"
    )

    logger.info("\n=== median_incomeのパーセンタイル変換 ===")
    flattened_income = percentile_transform(housing["median_income"])
    percentile_path = plot_percentile_income(flattened_income)
    logger.info(f"保存しました: {percentile_path}")
    logger.info(
        "1パーセンタイル未満は1、99パーセンタイル超は100にまとめられるため、"
        "分布はほぼ一様になり、値の範囲は1〜100に収まる。"
    )

    logger.info("\n=== housing_median_ageの年齢類似度（RBFカーネル） ===")
    age_similarities = rbf_age_similarity(housing["housing_median_age"])
    for gamma, similarity in age_similarities.items():
        logger.info(f"gamma={gamma}: 先頭5件 = {similarity[:5].round(3)}")
    age_similarity_path = plot_age_similarity(housing["housing_median_age"])
    logger.info(f"保存しました: {age_similarity_path}")
    logger.info(
        "gammaが大きいほど基準値（35歳）付近だけが高い類似度を持つ狭いピークになり、"
        "gammaが小さいほど遠い年齢まで緩やかに類似度が残る。"
    )

    logger.info("\n=== サンフランシスコとの地理的類似度（RBFカーネル） ===")
    sf_similarity = rbf_geo_similarity_sf(housing)
    logger.info(f"先頭5件 = {sf_similarity[:5].round(4)}")
    logger.info(
        "緯度・経度をそのまま数値として使う代わりに、特定地点への近さを"
        "連続値の類似度特徴量として表現できる。"
    )

    logger.info("\n=== TransformedTargetRegressorによる目的変数の変換 ===")
    predictions = compare_target_scaling(housing, housing_labels)
    logger.info(f"手動でinverse_transformした予測値: {predictions['manual'].round(1)}")
    logger.info(f"TransformedTargetRegressorの予測値: {predictions['wrapped'].round(1)}")
    logger.info(f"両者が一致するか: {np.allclose(predictions['manual'], predictions['wrapped'])}")
    logger.info(
        "目的変数を標準化して学習しても、TransformedTargetRegressorが"
        "predict()の内部で自動的に元の単位へ逆変換してくれるため、"
        "手動でtarget_scaler.inverse_transform()を呼ぶ場合と同じ予測値になる。"
    )


if __name__ == "__main__":
    main()
