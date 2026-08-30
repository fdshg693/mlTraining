"""単純ランダム分割 vs median_income層化分割を、複数のrandom_stateにわたって
テストRMSEの箱ひげ図で比較する実験。

02_split_data.pyでは1つのrandom_state（=42）についてincome_cat比率のズレを
確認した。ここでは01の`split_seed_sweep.py`と同じ発想で、複数のシードに対して
分割方法（単純ランダム vs 層化）ごとにモデルを訓練・評価し、テストRMSEの
ブレそのものを比較する。

【実行前に予想】住宅データ（20,640行）は01のlifesat（27行）よりはるかに
大きいので、シードによるテストRMSEのブレは小さくなるはず。層化 vs ランダムの
差も、01でincome_cat比率のズレが顕著だったほど劇的には出ない可能性がある——
「データ量が増えるとシードのブレはどう縮むか」を数値で確認する。
"""

from importlib import import_module
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

PROJECT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "learn"))
from common import TARGET_COLUMN, build_full_preprocessing  # noqa: E402
from data_produce.util.logging_config import setup_logger  # noqa: E402

load_housing_data = import_module("01_load_and_inspect").load_housing_data
add_income_cat = import_module("02_split_data").add_income_cat

MODEL_RANDOM_STATE = 42  # モデル側のrandom_state（分割のシードとは独立に固定）
SEEDS = range(10)
TEST_SIZE = 0.2
N_ESTIMATORS = 100


def build_model() -> Pipeline:
    """前処理とRandomForestRegressorを組み合わせたパイプラインを作る。"""

    return make_pipeline(
        build_full_preprocessing(),
        RandomForestRegressor(
            n_estimators=N_ESTIMATORS, random_state=MODEL_RANDOM_STATE, n_jobs=-1
        ),
    )


def split_random(
    housing_with_cat: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """income_catを無視した単純ランダム分割。"""

    train_set, test_set = train_test_split(
        housing_with_cat, test_size=TEST_SIZE, random_state=seed
    )
    return train_set.drop(columns=["income_cat"]), test_set.drop(columns=["income_cat"])


def split_stratified(
    housing_with_cat: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """income_catで層化した分割。"""

    train_set, test_set = train_test_split(
        housing_with_cat,
        test_size=TEST_SIZE,
        stratify=housing_with_cat["income_cat"],
        random_state=seed,
    )
    return train_set.drop(columns=["income_cat"]), test_set.drop(columns=["income_cat"])


def evaluate_split(train_set: pd.DataFrame, test_set: pd.DataFrame) -> float:
    """train_set/test_setでモデルを訓練し、テストRMSEを返す。"""

    X_train = train_set.drop(columns=[TARGET_COLUMN])
    y_train = train_set[TARGET_COLUMN]
    X_test = test_set.drop(columns=[TARGET_COLUMN])
    y_test = test_set[TARGET_COLUMN]

    model = build_model()
    model.fit(X_train, y_train)
    return root_mean_squared_error(y_test, model.predict(X_test))


def compute_rmse_sweep(housing_with_cat: pd.DataFrame) -> dict[str, np.ndarray]:
    """分割方法ごとに、シードを振ってテストRMSEを集計する。"""

    strategies = {"random": split_random, "stratified": split_stratified}
    results = {}
    for name, split_fn in strategies.items():
        scores = []
        for seed in SEEDS:
            train_set, test_set = split_fn(housing_with_cat, seed)
            rmse = evaluate_split(train_set, test_set)
            scores.append(rmse)
            logger.info(f"{name:>10} split, seed={seed}: テストRMSE={rmse:,.0f}")
        results[name] = np.array(scores)
    return results


def plot_sweep(results: dict[str, np.ndarray], outputs_dir: Path) -> Path:
    """分割方法 vs テストRMSEを箱ひげ図で描く。"""

    labels = list(results.keys())
    data = [results[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=labels)
    ax.set_xlabel("split strategy")
    ax.set_ylabel("test RMSE")
    ax.set_title(
        f"test RMSE by split strategy (random_state=0-{len(SEEDS) - 1}, "
        f"n={len(SEEDS)} runs)"
    )
    ax.grid(True)

    path = outputs_dir / "split_strategy_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    setup_logger(Path(__file__).stem)
    outputs_dir = Path(__file__).resolve().parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    housing = load_housing_data()
    housing_with_cat = add_income_cat(housing)
    logger.info(f"データセット全体: {housing.shape[0]:,}行")

    logger.info(f"\n=== 分割方法×シード（0〜{len(SEEDS) - 1}）ごとのテストRMSE ===")
    results = compute_rmse_sweep(housing_with_cat)

    for name, scores in results.items():
        logger.info(
            f"{name:>10}: テストRMSE 平均={scores.mean():,.0f} "
            f"標準偏差={scores.std():,.0f} min={scores.min():,.0f} max={scores.max():,.0f}"
        )

    path = plot_sweep(results, outputs_dir)
    logger.info(f"保存しました: {path}")

    random_scores, strat_scores = results["random"], results["stratified"]
    random_std, strat_std = random_scores.std(), strat_scores.std()
    random_mean, strat_mean = random_scores.mean(), strat_scores.mean()
    logger.info(
        f"\n標準偏差はrandom={random_std:,.0f} / stratified={strat_std:,.0f}。"
        f"{'層化の方がシードによるブレが小さい' if strat_std < random_std else '単純ランダムの方がブレが小さい、または同程度'}。"
    )
    logger.info(
        f"平均テストRMSEはrandom={random_mean:,.0f} / stratified={strat_mean:,.0f}"
        f"（差={random_mean - strat_mean:,.0f}）。"
    )


if __name__ == "__main__":
    main()
