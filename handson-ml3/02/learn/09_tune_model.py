"""前処理込みのfull_pipelineに対して、Grid SearchとRandomized Searchで
RandomForestRegressorのハイパーパラメータを探索する。

前処理（07_preprocessing_pipeline.py相当）はcommon.pyのbuild_full_preprocessing()
を再利用し、このファイルでは探索そのものに集中する。08_compare_models.pyで
random_forestが有力だったことを受け、ここではrandom_forestだけを深掘りする。
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from scipy.stats import expon, geom, loguniform, randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_DIR))
from common import build_full_preprocessing, load_features_and_labels
from data_produce.util.file_io import save_figure
from data_produce.util.logging_config import setup_logger

CV_FOLDS = 3
RANDOM_STATE = 42
SCORE_COLS = ["split0", "split1", "split2", "mean_test_rmse"]


def build_full_pipeline() -> Pipeline:
    """前処理とRandomForestRegressorを1本のパイプラインへまとめる。"""

    return Pipeline([
        ("preprocessing", build_full_preprocessing()),
        ("random_forest", RandomForestRegressor(random_state=RANDOM_STATE)),
    ])


def run_grid_search(
    full_pipeline: Pipeline, housing: pd.DataFrame, housing_labels: pd.Series
) -> tuple[GridSearchCV, float]:
    """geoクラスタ数とmax_featuresの組み合わせを総当たりで探索する。"""

    # 辞書のリストは複数の探索空間として扱われ、各配列の全組み合わせを試す。
    # 1つ目は3 x 3 = 9通り、2つ目は2 x 3 = 6通りで、合計15通りになる。
    # 配列の長さを揃える必要はないが、(n_clusters=10, max_features=6/8)は重複する。
    param_grid = [
        {"preprocessing__geo__n_clusters": [5, 8, 10],
         "random_forest__max_features": [4, 6, 8]},
        {"preprocessing__geo__n_clusters": [10, 15],
         "random_forest__max_features": [6, 8, 10]},
    ]
    grid_search = GridSearchCV(
        full_pipeline, param_grid, cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error",
    )
    start = time.perf_counter()
    grid_search.fit(housing, housing_labels)
    elapsed = time.perf_counter() - start
    return grid_search, elapsed


def run_randomized_search(
    full_pipeline: Pipeline,
    housing: pd.DataFrame,
    housing_labels: pd.Series,
    n_iter: int = 10,
) -> tuple[RandomizedSearchCV, float]:
    """geoクラスタ数とmax_featuresをrandintの分布から乱数探索する。"""

    param_distribs = {
        "preprocessing__geo__n_clusters": randint(low=3, high=50),
        "random_forest__max_features": randint(low=2, high=20),
    }
    rnd_search = RandomizedSearchCV(
        full_pipeline, param_distributions=param_distribs, n_iter=n_iter,
        cv=CV_FOLDS, scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
    )
    start = time.perf_counter()
    rnd_search.fit(housing, housing_labels)
    elapsed = time.perf_counter() - start
    return rnd_search, elapsed


def summarize_cv_results(search: GridSearchCV | RandomizedSearchCV) -> pd.DataFrame:
    """cv_results_から、探索した組み合わせとfold別・平均のRMSEだけを抜き出す。"""

    # cv_results_は、GridSearchCV/RandomizedSearchCVをfitした後に生成される辞書属性で、
    # 試した各パラメータの組み合わせを1行として、fold別スコア・平均/標準偏差スコア・
    # 順位（rank_test_score）・fit/predictの所要時間などをまとめて持つ（sklearn公式の探索結果ログ）。
    # 列が多く生の辞書のままでは扱いにくいため、DataFrame化した上でスコアの良い順（降順）に並べ替える。
    cv_res = pd.DataFrame(search.cv_results_)
    cv_res = cv_res.sort_values(by="mean_test_score", ascending=False)
    # 大量の列の中から、見たい列（探索したパラメータ2つ・fold別スコア3つ・平均スコア）だけを抜き出す。
    # 列名の"param_"接頭辞はGridSearchCV/RandomizedSearchCVが自動で付与するもの。
    cv_res = cv_res[[
        "param_preprocessing__geo__n_clusters",
        "param_random_forest__max_features",
        "split0_test_score", "split1_test_score", "split2_test_score",
        "mean_test_score",
    ]]
    # 上で選んだ6列に、短く読みやすい列名を付け直す（SCORE_COLS = ["split0", "split1", "split2", "mean_test_rmse"]）。
    cv_res.columns = ["n_clusters", "max_features"] + SCORE_COLS
    # scoring="neg_root_mean_squared_error"により、スコアはRMSEの符号を反転した値（マイナス）で入っている。
    # 符号を戻して正のRMSEにし、見やすいよう整数に丸める。
    cv_res[SCORE_COLS] = -cv_res[SCORE_COLS].round().astype(np.int64)
    # sort_valuesで元のインデックスが飛び飛びのままなので、0始まりの連番に振り直す。
    return cv_res.reset_index(drop=True)


def plot_sampling_distributions(outputs_dir: str | Path | None = None) -> Path:
    """randint/uniform/geom/expon/loguniformの形状を1枚の図にまとめて保存する。

    randint・uniformは「範囲内のどの値も同じくらいありそう」なとき、
    geom・exponは「だいたいこの規模」という中心値があるとき、
    loguniformは「桁のスケールすら見当がつかない」ときに使う分布。
    """

    fig, axs = plt.subplots(2, 3, figsize=(12, 6))

    xs1 = np.arange(0, 7 + 1)
    axs[0, 0].bar(xs1, randint(0, 7 + 1).pmf(xs1))
    axs[0, 0].set_title("randint(0, 8)")
    axs[0, 0].set_ylabel("Probability")

    xs2 = np.linspace(0, 7, 500)
    axs[0, 1].fill_between(xs2, uniform(0, 7).pdf(xs2))
    axs[0, 1].set_title("uniform(0, 7)")

    xs3 = np.arange(0, 7 + 1)
    axs[0, 2].bar(xs3, geom(0.5).pmf(xs3))
    axs[0, 2].set_title("geom(0.5)")

    xs4 = np.linspace(0, 7, 500)
    axs[1, 0].fill_between(xs4, expon(scale=1).pdf(xs4))
    axs[1, 0].set_title("expon(scale=1)")
    axs[1, 0].set_ylabel("PDF")
    axs[1, 0].set_xlabel("Hyperparameter value")

    xs5 = np.linspace(0.001, 1000, 500)
    axs[1, 1].fill_between(xs5, loguniform(0.001, 1000).pdf(xs5))
    axs[1, 1].set_title("loguniform(0.001, 1000)")
    axs[1, 1].set_xlabel("Hyperparameter value")

    log_xs = np.linspace(np.log(0.001), np.log(1000), 500)
    axs[1, 2].fill_between(log_xs, uniform(np.log(0.001), np.log(1000)).pdf(log_xs))
    axs[1, 2].set_title("log(X), X ~ loguniform")
    axs[1, 2].set_xlabel("Log of hyperparameter value")

    fig.tight_layout()
    if outputs_dir is None:
        path = save_figure(fig, "sampling_distributions.png", dpi=300)
    else:
        path = save_figure(fig, "sampling_distributions.png", outputs_dir, dpi=300)
    plt.close(fig)
    return path


def main() -> None:
    setup_logger(Path(__file__).stem)
    housing, housing_labels = load_features_and_labels()
    logger.info(f"訓練セット（特徴量）: {housing.shape[0]:,}行 x {housing.shape[1]}列")

    full_pipeline = build_full_pipeline()
    logger.info("\n=== full_pipelineのパラメータ名（一部） ===")
    param_keys = sorted(full_pipeline.get_params().keys())
    logger.info(f"パラメータ数: {len(param_keys)}")
    logger.info("先頭10件: " + ", ".join(param_keys[:10]))
    logger.info(
        "'preprocessing__geo__n_clusters'のように"
        "ステップ名__パラメータ名（さらにネストしていれば__を重ねる）で、"
        "パイプライン内部の変換器・推定器のパラメータも探索対象にできる。"
    )

    logger.info(f"\n=== Grid Search（cv={CV_FOLDS}） ===")
    grid_search, grid_elapsed = run_grid_search(full_pipeline, housing, housing_labels)
    logger.info(f"実行時間: {grid_elapsed:.1f}秒")
    logger.info(f"最良パラメータ: {grid_search.best_params_}")
    logger.info(f"最良RMSE: {-grid_search.best_score_:,.0f}")
    grid_cv_res = summarize_cv_results(grid_search)
    logger.info(f"探索結果（上位から）:\n{grid_cv_res}")
    for param_name, candidates in [
        ("n_clusters", [5, 8, 10, 15]),
        ("max_features", [4, 6, 8, 10]),
    ]:
        best_value = grid_cv_res.iloc[0][param_name]
        if best_value == max(candidates):
            logger.info(
                f"{param_name}の最良値({best_value})が探索範囲の上端に張り付いている。"
                "さらに大きい値も試す価値がある。"
            )

    n_iter = 10
    logger.info(f"\n=== Randomized Search（n_iter={n_iter}, cv={CV_FOLDS}） ===")
    rnd_search, rnd_elapsed = run_randomized_search(
        full_pipeline, housing, housing_labels, n_iter=n_iter
    )
    logger.info(f"実行時間: {rnd_elapsed:.1f}秒（{n_iter}件 x {CV_FOLDS}-fold = {n_iter * CV_FOLDS}回学習）")
    logger.info(f"最良パラメータ: {rnd_search.best_params_}")
    logger.info(f"最良RMSE: {-rnd_search.best_score_:,.0f}")
    rnd_cv_res = summarize_cv_results(rnd_search)
    logger.info(f"探索結果（上位から）:\n{rnd_cv_res}")
    logger.info(
        f"random_state={RANDOM_STATE}に固定しているため、同じ環境・同じscikit-learn"
        "バージョンであれば再実行しても同じ組み合わせが同じ順序で試される。"
    )

    logger.info("\n=== Grid Search と Randomized Search の比較 ===")
    logger.info(
        f"Grid Search: {len(grid_search.cv_results_['params'])}通り x {CV_FOLDS}-fold, "
        f"{grid_elapsed:.1f}秒, 最良RMSE={-grid_search.best_score_:,.0f}"
    )
    logger.info(
        f"Randomized Search: {n_iter}通り x {CV_FOLDS}-fold, "
        f"{rnd_elapsed:.1f}秒, 最良RMSE={-rnd_search.best_score_:,.0f}"
    )
    logger.info(
        "Grid Searchは候補が少ない場合に全組み合わせを漏れなく比較できるが、"
        "パラメータが増えると組み合わせ数が指数的に増える。"
        "Randomized Searchは試行回数を固定したまま連続的な探索空間を扱えるため、"
        "広い範囲やパラメータ数が多い場合に効率が良くなりやすい。"
    )

    logger.info("\n=== 乱数探索で使う確率分布の形 ===")
    dist_path = plot_sampling_distributions()
    logger.info(f"保存しました: {dist_path}")
    logger.info(
        "randint/uniformは範囲内のどの値も同程度ありそうなとき、"
        "geom/exponはおおよその規模（スケール）だけ分かっているとき、"
        "loguniformは値の桁数（スケール自体）に見当がつかないときに向いている。"
    )


if __name__ == "__main__":
    main()
