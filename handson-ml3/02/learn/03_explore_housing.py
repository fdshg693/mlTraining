"""訓練セットだけを使い、地理的分布・相関・組み合わせ特徴量を探索する。"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from pandas.plotting import scatter_matrix

PROJECT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_DIR))
from data_produce.util.file_io import data_path, read_csv, save_figure
from data_produce.util.logging_config import setup_logger

HOUSING_TRAIN_CSV = "housing_train.csv"
CORRELATION_ATTRIBUTES = [
    "median_house_value",
    "median_income",
    "total_rooms",
    "housing_median_age",
]


def load_train_set() -> pd.DataFrame:
    """02_split_data.pyが保存した層化分割済みの訓練セットを読み込む。"""

    if not data_path(HOUSING_TRAIN_CSV).is_file():
        raise FileNotFoundError(
            f"{HOUSING_TRAIN_CSV}が見つかりません。"
            "先にlearn/02_split_data.pyを実行してください。"
        )
    return read_csv(HOUSING_TRAIN_CSV)


def _save(figure, file_name: str, outputs_dir: str | Path | None) -> Path:
    if outputs_dir is None:
        return save_figure(figure, file_name, dpi=300)
    return save_figure(figure, file_name, outputs_dir, dpi=300)


def plot_geo_scatter_density(
    housing: pd.DataFrame,
    outputs_dir: str | Path | None = None,
) -> Path:
    """緯度・経度の散布図を透明度付きで描き、地区の密集具合を確認する。"""

    axis = housing.plot(
        kind="scatter",
        x="longitude",
        y="latitude",
        alpha=0.2,
        grid=True,
        figsize=(8, 6),
    )
    figure = axis.get_figure()
    figure.tight_layout()
    path = _save(figure, "geo_scatter_density.png", outputs_dir)
    plt.close(figure)
    return path


def plot_geo_scatter_population_price(
    housing: pd.DataFrame,
    outputs_dir: str | Path | None = None,
) -> Path:
    """人口を点の大きさ、住宅価格を色に割り当てた地理散布図を保存する。"""

    axis = housing.plot(
        kind="scatter",
        x="longitude",
        y="latitude",
        grid=True,
        s=housing["population"] / 100,
        label="population",
        c="median_house_value",
        cmap="jet",
        colorbar=True,
        legend=True,
        sharex=False,
        figsize=(10, 7),
    )
    figure = axis.get_figure()
    figure.tight_layout()
    path = _save(figure, "geo_scatter_population_price.png", outputs_dir)
    plt.close(figure)
    return path


def correlations_with_target(
    housing: pd.DataFrame,
    target: str = "median_house_value",
) -> pd.Series:
    """数値列とtargetとの相関係数を、絶対値が大きい順に並べて返す。"""

    corr_matrix = housing.corr(numeric_only=True)
    return corr_matrix[target].sort_values(ascending=False)


def plot_scatter_matrix(
    housing: pd.DataFrame,
    attributes: list[str] = CORRELATION_ATTRIBUTES,
    outputs_dir: str | Path | None = None,
) -> Path:
    """指定した特徴量同士の散布図行列を保存する。"""

    axes = scatter_matrix(housing[attributes], figsize=(12, 8))
    figure = axes.flat[0].get_figure()
    figure.tight_layout()
    path = _save(figure, "scatter_matrix_plot.png", outputs_dir)
    plt.close(figure)
    return path


def plot_income_vs_value(
    housing: pd.DataFrame,
    outputs_dir: str | Path | None = None,
) -> Path:
    """median_incomeとmedian_house_valueの散布図を保存する。"""

    axis = housing.plot(
        kind="scatter",
        x="median_income",
        y="median_house_value",
        alpha=0.1,
        grid=True,
    )
    figure = axis.get_figure()
    figure.tight_layout()
    path = _save(figure, "income_vs_house_value_scatter.png", outputs_dir)
    plt.close(figure)
    return path


def find_capped_values(
    housing: pd.DataFrame,
    column: str,
    top_n: int = 5,
) -> pd.Series:
    """出現回数が多い値を調べ、上限打ち切りらしき値を見つける。

    散布図に現れる水平な点の列は、多くの地区が同じ値を持つことを示す。
    これは自然な分布では起きにくく、収集・集計段階での打ち切りを疑う手がかりになる。
    """

    return housing[column].value_counts().head(top_n)


def add_combined_attributes(housing: pd.DataFrame) -> pd.DataFrame:
    """総数だけでは分かりにくい比率をコピー上に追加する。"""

    housing = housing.copy()
    housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["people_per_house"] = housing["population"] / housing["households"]
    return housing


def main() -> None:
    setup_logger(Path(__file__).stem)
    housing = load_train_set()
    logger.info(f"訓練セット: {housing.shape[0]:,}行 x {housing.shape[1]}列")

    logger.info("\n=== 地理的分布（密集具合） ===")
    density_path = plot_geo_scatter_density(housing)
    logger.info(f"保存しました: {density_path}")

    logger.info("\n=== 地理的分布（人口・住宅価格） ===")
    population_price_path = plot_geo_scatter_population_price(housing)
    logger.info(f"保存しました: {population_price_path}")

    logger.info("\n=== median_house_valueとの相関（元の特徴量） ===")
    base_corr = correlations_with_target(housing)
    logger.info(f"\n{base_corr.to_string()}")

    logger.info("\n=== 散布図行列 ===")
    scatter_matrix_path = plot_scatter_matrix(housing)
    logger.info(f"保存しました: {scatter_matrix_path}")

    logger.info("\n=== median_income vs median_house_value ===")
    income_value_path = plot_income_vs_value(housing)
    logger.info(f"保存しました: {income_value_path}")

    logger.info("\n=== median_house_valueの上限打ち切り候補 ===")
    capped_values = find_capped_values(housing, "median_house_value")
    logger.info(f"\n{capped_values.to_string()}")
    logger.info(
        "出現回数が突出して多い値は、実際の分布ではなく上限で打ち切られた値である"
        "可能性が高い。上限付近の地区をそのまま学習に使うか、除外するかは"
        "モデルの用途に応じて判断が必要。"
    )

    logger.info("\n=== 組み合わせ特徴量の追加 ===")
    housing_with_combined = add_combined_attributes(housing)
    combined_corr = correlations_with_target(housing_with_combined)
    logger.info(f"\n{combined_corr.to_string()}")

    logger.info(
        "\nrooms_per_houseはtotal_roomsより強い相関を持ち、世帯あたりの部屋数が"
        "価格と結び付きやすいことを示す。bedrooms_ratioは負の相関を持ち、"
        "寝室の割合が低い（＝居室が多い）ほど価格が高い傾向を示す。"
        "people_per_houseはpopulationよりわずかに強い相関を持つ。"
        "3つとも既存の列の比率から作られており、目的変数を使っていないため"
        "リーケージはないが、実際にモデルへ渡すかは前処理段階で再検証する。"
    )


if __name__ == "__main__":
    main()
