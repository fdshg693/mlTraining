"""住宅価格データを訓練・テストセットへ分割し、分割方法ごとの代表性を比較する。"""

from importlib import import_module
from pathlib import Path
import sys
from zlib import crc32

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_DIR))
from data_produce.util.file_io import save_figure, write_csv
from data_produce.util.logging_config import setup_logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_housing_data = import_module("01_load_and_inspect").load_housing_data


def shuffle_and_split_data(
    data: pd.DataFrame,
    test_ratio: float,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """np.random.permutationで単純にシャッフルして分割する。

    シードを固定しない場合、実行のたびにテストセットの中身が変わる。
    """

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def is_id_in_test_set(identifier: int, test_ratio: float) -> bool:
    return crc32(np.int64(identifier)) < test_ratio * 2**32


def split_data_with_id_hash(
    data: pd.DataFrame,
    test_ratio: float,
    id_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """IDのハッシュ値で分割する。

    データが後から追加されても、既存の行はIDが変わらない限り
    同じ側（訓練/テスト）に留まり続ける。
    """

    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    # ~in_test_set はブール値を反転させ、テストセットに含まれない行（=訓練セット）を選ぶ
    return data.loc[~in_test_set], data.loc[in_test_set]


def add_income_cat(housing: pd.DataFrame) -> pd.DataFrame:
    """median_incomeを5カテゴリに分けたincome_cat列を追加したコピーを返す。"""

    housing = housing.copy()
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    return housing


def plot_income_cat(
    housing: pd.DataFrame,
    outputs_dir: str | Path | None = None,
) -> Path:
    """income_catごとの件数を棒グラフとして保存する。"""

    axis = housing["income_cat"].value_counts().sort_index().plot.bar(
        rot=0, grid=True, figsize=(8, 5)
    )
    axis.set_xlabel("Income category")
    axis.set_ylabel("Number of districts")
    figure = axis.get_figure()
    figure.tight_layout()

    if outputs_dir is None:
        path = save_figure(figure, "housing_income_cat_bar_plot.png", dpi=300)
    else:
        path = save_figure(
            figure, "housing_income_cat_bar_plot.png", outputs_dir, dpi=300
        )
    plt.close(figure)
    return path


def income_cat_proportions(data: pd.DataFrame) -> pd.Series:
    return data["income_cat"].value_counts() / len(data)


def compare_split_strategies(
    housing_with_cat: pd.DataFrame,
    strat_test_set: pd.DataFrame,
    random_test_set: pd.DataFrame,
) -> pd.DataFrame:
    """全体・層化分割・単純ランダム分割のincome_cat比率とその誤差(%)をまとめる。"""

    compare_props = pd.DataFrame(
        {
            "Overall %": income_cat_proportions(housing_with_cat),
            "Stratified %": income_cat_proportions(strat_test_set),
            "Random %": income_cat_proportions(random_test_set),
        }
    ).sort_index()
    compare_props.index.name = "Income Category"
    compare_props["Strat. Error %"] = (
        compare_props["Stratified %"] / compare_props["Overall %"] - 1
    )
    compare_props["Rand. Error %"] = (
        compare_props["Random %"] / compare_props["Overall %"] - 1
    )
    return (compare_props * 100).round(2)


def save_splits(
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    data_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """層化分割した訓練・テストセットをCSVで保存し、以降のスクリプトから再利用できるようにする。"""

    if data_dir is None:
        train_path = write_csv(train_set, "housing_train.csv", index=False)
        test_path = write_csv(test_set, "housing_test.csv", index=False)
    else:
        train_path = write_csv(train_set, "housing_train.csv", data_dir, index=False)
        test_path = write_csv(test_set, "housing_test.csv", data_dir, index=False)
    return train_path, test_path


def main() -> None:
    setup_logger(Path(__file__).stem)
    housing = load_housing_data()

    logger.info("=== 単純シャッフル分割（np.random.permutation） ===")
    train_set, test_set = shuffle_and_split_data(housing, 0.2, seed=42)
    logger.info(f"train: {len(train_set):,}行 / test: {len(test_set):,}行")

    logger.info("\n=== シードを変えるとテストセットが変わることの確認 ===")
    _, test_set_seed1 = shuffle_and_split_data(housing, 0.2, seed=1)
    _, test_set_seed2 = shuffle_and_split_data(housing, 0.2, seed=2)
    overlap = len(set(test_set_seed1.index) & set(test_set_seed2.index))
    logger.info(
        f"seed=1とseed=2のテストセットが共有する行数: "
        f"{overlap} / {len(test_set_seed1)}"
    )

    logger.info("\n=== 行番号IDのハッシュ分割 ===")
    housing_with_id = housing.reset_index()  # 「index」という名前の列が追加される
    id_train_set, id_test_set = split_data_with_id_hash(
        housing_with_id, 0.2, "index"
    )
    logger.info(f"train: {len(id_train_set):,}行 / test: {len(id_test_set):,}行")

    logger.info("\n=== 緯度経度から作ったIDのハッシュ分割 ===")
    housing_with_id["geo_id"] = (
        housing["longitude"] * 1000 + housing["latitude"]
    )
    geo_train_set, geo_test_set = split_data_with_id_hash(
        housing_with_id, 0.2, "geo_id"
    )
    logger.info(f"train: {len(geo_train_set):,}行 / test: {len(geo_test_set):,}行")
    logger.info(
        "行番号IDはデータが末尾に追加される前提でしか安定しないが、"
        "緯度経度から作るIDは行の並びが変わっても同じ地区が同じ側に残りやすい。"
    )

    logger.info("\n=== train_test_split(random_state=42)による標準的な分割 ===")
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    logger.info(f"train: {len(train_set):,}行 / test: {len(test_set):,}行")
    logger.info(
        "total_bedroomsの欠損数(test): "
        f"{test_set['total_bedrooms'].isnull().sum()}"
    )

    logger.info("\n=== income_catによる層化分割の準備 ===")
    housing_with_cat = add_income_cat(housing)
    income_cat_path = plot_income_cat(housing_with_cat)
    logger.info(f"income_catの分布を保存しました: {income_cat_path}")

    # n_splits: train/testの分割を何セット生成するか(交差検証のfold数のようなもの)。ここでは異なる分割を10通り作る
    splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    strat_splits = [
        (housing_with_cat.iloc[train_index], housing_with_cat.iloc[test_index])
        # split()の第2引数に渡したhousing_with_cat["income_cat"]を層化キーとして使い、
        # 各income_catカテゴリの比率がtrain/testそれぞれで元データと同じになるように分割する
        for train_index, test_index in splitter.split(
            housing_with_cat, housing_with_cat["income_cat"]
        )
    ]
    logger.info(f"StratifiedShuffleSplitで作成した分割数: {len(strat_splits)}")

    strat_train_set, strat_test_set = train_test_split(
        housing_with_cat,
        test_size=0.2,
        stratify=housing_with_cat["income_cat"],
        random_state=42,
    )
    random_train_set, random_test_set = train_test_split(
        housing_with_cat, test_size=0.2, random_state=42
    )

    logger.info("\n=== 層化分割と単純ランダム分割のincome_cat比率比較 ===")
    compare_props = compare_split_strategies(
        housing_with_cat, strat_test_set, random_test_set
    )
    logger.info(f"\n{compare_props.to_string()}")

    logger.info("\n=== income_cat列を削除し、コピーとして確定 ===")
    strat_train_set = strat_train_set.drop(columns=["income_cat"]).copy()
    strat_test_set = strat_test_set.drop(columns=["income_cat"]).copy()
    logger.info(
        f"strat_train_set: {strat_train_set.shape}, "
        f"strat_test_set: {strat_test_set.shape}"
    )

    train_path, test_path = save_splits(strat_train_set, strat_test_set)
    logger.info(f"\n分割済みデータを保存しました: {train_path}")
    logger.info(f"分割済みデータを保存しました: {test_path}")


if __name__ == "__main__":
    main()
