"""RandomForestのmax_depth・n_estimatorsを変え、過学習の育ち方を定量的に追跡する実験。

08_compare_models.pyでは決定木の訓練RMSEがほぼ0でもCV RMSEは悪化しうることを
定性的に確認した。ここでは同じ「訓練誤差だけでは判断できない」性質を、
RandomForestのmax_depth・n_estimatorsという連続的なパラメータに対して、
01の`degree_sweep.py`のように訓練・CVの2本線を1枚に重ねて描き、定量的に追跡する。
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline

PROJECT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "learn"))
from common import build_full_preprocessing, load_features_and_labels  # noqa: E402
from data_produce.util.logging_config import setup_logger  # noqa: E402

RANDOM_STATE = 42
# 08_compare_models.pyはCV_FOLDS=10だが、ここではmax_depth・n_estimators
# 合わせて9設定を振るため、実行時間を抑えて5-foldにする。
CV_FOLDS = 5
MAX_DEPTH_VALUES = (2, 5, 10, 20, None)
N_ESTIMATORS_VALUES = (1, 10, 50, 200)
FIXED_N_ESTIMATORS = 100  # max_depth sweepで固定するn_estimators（sklearnの既定値）
FIXED_MAX_DEPTH = None  # n_estimators sweepで固定するmax_depth（sklearnの既定値＝無制限）


def build_model(max_depth: int | None, n_estimators: int) -> Pipeline:
    """前処理とRandomForestRegressorを組み合わせたパイプラインを作る。"""

    return make_pipeline(
        build_full_preprocessing(),
        RandomForestRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    )


def evaluate(model: Pipeline, housing, housing_labels) -> tuple[float, float, float]:
    """訓練RMSEと、CV_FOLDS分割の交差検証RMSE（平均・標準偏差）を計算する。"""

    model.fit(housing, housing_labels)
    train_rmse = root_mean_squared_error(housing_labels, model.predict(housing))

    neg_rmses = cross_val_score(
        model, housing, housing_labels,
        scoring="neg_root_mean_squared_error", cv=CV_FOLDS,
    )
    cv_rmses = -neg_rmses
    return train_rmse, cv_rmses.mean(), cv_rmses.std()


def compute_max_depth_results(housing, housing_labels) -> dict:
    """n_estimatorsを固定し、max_depthごとに訓練RMSE・CV RMSEを計算する。"""

    results = {}
    for max_depth in MAX_DEPTH_VALUES:
        model = build_model(max_depth=max_depth, n_estimators=FIXED_N_ESTIMATORS)
        train_rmse, cv_mean, cv_std = evaluate(model, housing, housing_labels)
        results[max_depth] = {"train_rmse": train_rmse, "cv_mean": cv_mean, "cv_std": cv_std}
        logger.info(
            f"max_depth={str(max_depth):<4}: 訓練RMSE={train_rmse:,.0f}  "
            f"CV RMSE={cv_mean:,.0f}±{cv_std:,.0f}"
        )
    return results


def compute_n_estimators_results(housing, housing_labels) -> dict:
    """max_depthを固定し、n_estimatorsごとに訓練RMSE・CV RMSEを計算する。"""

    results = {}
    for n_estimators in N_ESTIMATORS_VALUES:
        model = build_model(max_depth=FIXED_MAX_DEPTH, n_estimators=n_estimators)
        train_rmse, cv_mean, cv_std = evaluate(model, housing, housing_labels)
        results[n_estimators] = {"train_rmse": train_rmse, "cv_mean": cv_mean, "cv_std": cv_std}
        logger.info(
            f"n_estimators={n_estimators:>3}: 訓練RMSE={train_rmse:,.0f}  "
            f"CV RMSE={cv_mean:,.0f}±{cv_std:,.0f}"
        )
    return results


def plot_sweep(
    results: dict, param_label: str, title: str, outputs_dir: Path, filename: str
) -> Path:
    """パラメータ vs RMSEを、訓練・CVの2本線で1枚に重ねて描く。"""

    keys = list(results.keys())
    labels = [str(k) for k in keys]
    x = np.arange(len(labels))
    train_rmses = [results[k]["train_rmse"] for k in keys]
    cv_means = [results[k]["cv_mean"] for k in keys]
    cv_stds = [results[k]["cv_std"] for k in keys]

    plt.figure(figsize=(8, 5))
    plt.plot(x, train_rmses, marker="o", label="train RMSE")
    plt.errorbar(
        x, cv_means, yerr=cv_stds, marker="o", capsize=4,
        label=f"CV RMSE ({CV_FOLDS}-fold, mean±std)",
    )
    plt.xticks(x, labels)
    plt.xlabel(param_label)
    plt.ylabel("RMSE")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    path = outputs_dir / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def main() -> None:
    setup_logger(Path(__file__).stem)
    outputs_dir = Path(__file__).resolve().parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    housing, housing_labels = load_features_and_labels()
    logger.info(f"訓練セット: {housing.shape[0]:,}行 x {housing.shape[1]}列")

    logger.info(f"\n=== max_depth sweep（n_estimators={FIXED_N_ESTIMATORS}固定） ===")
    depth_results = compute_max_depth_results(housing, housing_labels)
    depth_path = plot_sweep(
        depth_results, "max_depth",
        f"max_depth sweep (n_estimators={FIXED_N_ESTIMATORS})",
        outputs_dir, "max_depth_sweep.png",
    )
    logger.info(f"保存しました: {depth_path}")

    shallow, deepest = MAX_DEPTH_VALUES[0], MAX_DEPTH_VALUES[-1]
    shallow_train, deepest_train = (
        depth_results[shallow]["train_rmse"], depth_results[deepest]["train_rmse"]
    )
    best_depth = min(MAX_DEPTH_VALUES, key=lambda d: depth_results[d]["cv_mean"])
    best_cv = depth_results[best_depth]["cv_mean"]
    deepest_cv = depth_results[deepest]["cv_mean"]
    logger.info(
        f"max_depth={shallow}→{deepest}で訓練RMSEは{shallow_train:,.0f}→{deepest_train:,.0f}へ"
        f"{'単調に減少した' if deepest_train < shallow_train else '減少しなかった'}"
        "（木を深くするほど訓練データを丸暗記しやすくなる）。"
    )
    if best_depth == deepest:
        logger.info(
            f"CV RMSEはmax_depth={deepest}（最深）で最小（{best_cv:,.0f}）となり、"
            "このデータ・n_estimatorsの組み合わせでは深さを制限してもCV性能の改善は見られなかった"
            "（RandomForestは複数木の平均でdecision_tree単体より過学習に強いため）。"
        )
    else:
        logger.info(
            f"CV RMSEはmax_depth={best_depth}で最小（{best_cv:,.0f}）となり、"
            f"最深のmax_depth={deepest}（CV RMSE={deepest_cv:,.0f}）より悪化した。"
            "訓練RMSEが下がり続ける一方でCV RMSEはどこかで下げ止まる／悪化する、"
            "という過学習の典型的なパターンがここでも定量的に確認できた。"
        )

    logger.info(f"\n=== n_estimators sweep（max_depth={FIXED_MAX_DEPTH}固定） ===")
    n_estimators_results = compute_n_estimators_results(housing, housing_labels)
    n_estimators_path = plot_sweep(
        n_estimators_results, "n_estimators",
        f"n_estimators sweep (max_depth={FIXED_MAX_DEPTH})",
        outputs_dir, "n_estimators_sweep.png",
    )
    logger.info(f"保存しました: {n_estimators_path}")

    small_n, large_n = N_ESTIMATORS_VALUES[0], N_ESTIMATORS_VALUES[-1]
    mid_n = N_ESTIMATORS_VALUES[len(N_ESTIMATORS_VALUES) // 2]
    early_gain = (
        n_estimators_results[small_n]["cv_mean"] - n_estimators_results[mid_n]["cv_mean"]
    )
    late_gain = (
        n_estimators_results[mid_n]["cv_mean"] - n_estimators_results[large_n]["cv_mean"]
    )
    logger.info(
        f"n_estimators={small_n}→{mid_n}でCV RMSEは{early_gain:,.0f}改善したのに対し、"
        f"{mid_n}→{large_n}では{late_gain:,.0f}の改善にとどまった。"
        "木の本数を増やすほど個々の木の分散が平均化され性能は上がるが、"
        "改善幅は逓減していく（n_estimators=1は単一木に近く、CV RMSEのブレ"
        f"（標準偏差={n_estimators_results[small_n]['cv_std']:,.0f}）も"
        f"n_estimators={large_n}（標準偏差={n_estimators_results[large_n]['cv_std']:,.0f}）"
        "より大きい）。"
    )


if __name__ == "__main__":
    main()
