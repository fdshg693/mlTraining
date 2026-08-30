"""上限打ち切りされたmedian_house_value（$500,001）の扱いが、
テストRMSE・上限付近の予測誤差にどう影響するかを比較する実験。

背景: 元データの`median_house_value`には$500,001で頭打ちになっている行が
965件（$500,000ちょうどの27件を合わせると992件）あり、これまでの
learn/・exercise/では未対処のまま学習・評価に使ってきた。ここでは
上限付近（$500,000以上）の行を(a)そのまま含める、(b)訓練データから
削除する、(c)上限判定を分類器に任せてCAP_VALUEをそのまま出力する
（回帰による外挿は諦める）の3パターンで訓練し、テストセット全体・
上限付近のみ・非上限のみのRMSEを比較する。

【実行前に予想】(b)で上限行を訓練から除くと、非上限部分のRMSEは
（ノイズになっていた頭打ちラベルが減るため）多少改善するはず。だが
RandomForestは訓練データの値域を超えて外挿できないため、上限行に対する
予測は頭打ちの値そのものを一度も見ていない分、系統的に過小評価される
はず——つまり「全体RMSEの改善」は上限付近の予測能力を犠牲にした結果
ではないか。(c)は上限判定さえ当てられれば、その犠牲を払わずに
両立できるはず。
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import precision_score, recall_score, root_mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline

PROJECT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "learn"))
from common import HOUSING_TEST_CSV, build_full_preprocessing, load_features_and_labels  # noqa: E402
from data_produce.util.logging_config import setup_logger  # noqa: E402

RANDOM_STATE = 42
N_ESTIMATORS = 100
CAP_THRESHOLD = 500_000.0  # 上限付近と判定する閾値（PLAN.mdの例に合わせる）
CAP_VALUE = 500_001.0  # 元データで頭打ちになっている値（READMEで既知の仕様）


def build_regressor() -> Pipeline:
    """前処理とRandomForestRegressorを組み合わせたパイプラインを作る。"""

    return make_pipeline(
        build_full_preprocessing(),
        RandomForestRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
        ),
    )


def build_classifier() -> Pipeline:
    """前処理とRandomForestClassifierを組み合わせたパイプラインを作る（上限判定用）。"""

    return make_pipeline(
        build_full_preprocessing(),
        RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
        ),
    )


def run_pattern_as_is(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> np.ndarray:
    """(a) そのまま含める: 上限打ち切り行を含めて訓練する。"""

    model = build_regressor()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def run_pattern_drop(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_capped_mask: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """(b) 削除する: 上限打ち切り行を訓練データから除いて訓練する。"""

    model = build_regressor()
    model.fit(X_train[~train_capped_mask], y_train[~train_capped_mask])
    return model.predict(X_test)


def run_pattern_flag(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_capped_mask: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, Pipeline]:
    """(c) フラグ化する: 上限打ち切りかどうかを分類器で判定し、上限と判定した行は
    CAP_VALUEをそのまま出力、それ以外は非上限行のみで訓練した回帰器の予測を使う
    （目的変数の上限外挿は諦め、判定を分類器に任せる）。
    """

    classifier = build_classifier()
    classifier.fit(X_train, train_capped_mask)

    regressor = build_regressor()
    regressor.fit(X_train[~train_capped_mask], y_train[~train_capped_mask])

    predicted_capped = classifier.predict(X_test)
    y_pred = np.where(predicted_capped, CAP_VALUE, regressor.predict(X_test))
    return y_pred, classifier


def evaluate_by_subset(
    y_true: pd.Series, y_pred: np.ndarray, capped_mask: pd.Series
) -> dict[str, float]:
    """テストセット全体・上限行のみ・非上限行のみでRMSEを計算する。"""

    y_pred_series = pd.Series(y_pred, index=y_true.index)
    return {
        "all": root_mean_squared_error(y_true, y_pred_series),
        "non_capped": root_mean_squared_error(
            y_true[~capped_mask], y_pred_series[~capped_mask]
        ),
        "capped": root_mean_squared_error(
            y_true[capped_mask], y_pred_series[capped_mask]
        ),
    }


def plot_comparison(results: dict[str, dict[str, float]], outputs_dir: Path) -> Path:
    """パターン×テスト部分集合でRMSEをグループ棒グラフとして描く。"""

    patterns = list(results.keys())
    subsets = ["all", "non_capped", "capped"]
    subset_labels = {
        "all": "test (all)",
        "non_capped": "test (non-capped)",
        "capped": "test (capped only)",
    }

    x = np.arange(len(subsets))
    width = 0.8 / len(patterns)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, pattern in enumerate(patterns):
        values = [results[pattern][subset] for subset in subsets]
        offset = (i - (len(patterns) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=pattern)

    ax.set_xticks(x)
    ax.set_xticklabels([subset_labels[subset] for subset in subsets])
    ax.set_ylabel("test RMSE")
    ax.set_title(f"test RMSE by pattern and subset (cap threshold=${CAP_THRESHOLD:,.0f})")
    ax.legend()
    ax.grid(True, axis="y")

    path = outputs_dir / "capped_value_experiment.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    setup_logger(Path(__file__).stem)
    outputs_dir = Path(__file__).resolve().parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    X_train, y_train = load_features_and_labels()
    X_test, y_test = load_features_and_labels(HOUSING_TEST_CSV)

    train_capped_mask = y_train >= CAP_THRESHOLD
    test_capped_mask = y_test >= CAP_THRESHOLD
    logger.info(
        f"訓練: {len(y_train):,}行中 上限付近={train_capped_mask.sum():,}行"
        f"（{train_capped_mask.mean():.1%}）"
    )
    logger.info(
        f"テスト: {len(y_test):,}行中 上限付近={test_capped_mask.sum():,}行"
        f"（{test_capped_mask.mean():.1%}）"
    )

    results: dict[str, dict[str, float]] = {}

    logger.info("\n=== (a) そのまま含める ===")
    pred_as_is = run_pattern_as_is(X_train, y_train, X_test)
    results["as_is"] = evaluate_by_subset(y_test, pred_as_is, test_capped_mask)
    logger.info(
        f"RMSE all={results['as_is']['all']:,.0f} "
        f"non_capped={results['as_is']['non_capped']:,.0f} "
        f"capped={results['as_is']['capped']:,.0f}"
    )

    logger.info("\n=== (b) 削除する ===")
    pred_drop = run_pattern_drop(X_train, y_train, train_capped_mask, X_test)
    results["drop"] = evaluate_by_subset(y_test, pred_drop, test_capped_mask)
    logger.info(
        f"RMSE all={results['drop']['all']:,.0f} "
        f"non_capped={results['drop']['non_capped']:,.0f} "
        f"capped={results['drop']['capped']:,.0f}"
    )

    logger.info("\n=== (c) フラグ化する ===")
    pred_flag, classifier = run_pattern_flag(X_train, y_train, train_capped_mask, X_test)
    results["flag"] = evaluate_by_subset(y_test, pred_flag, test_capped_mask)
    predicted_capped_test = classifier.predict(X_test)
    precision = precision_score(
        test_capped_mask.astype(int), predicted_capped_test.astype(int)
    )
    recall = recall_score(
        test_capped_mask.astype(int), predicted_capped_test.astype(int)
    )
    logger.info(
        f"上限判定分類器: precision={precision:.2f} recall={recall:.2f}"
        "（適合率=上限と判定したうち実際に上限だった割合、"
        "再現率=実際の上限行のうち検出できた割合）"
    )
    logger.info(
        f"RMSE all={results['flag']['all']:,.0f} "
        f"non_capped={results['flag']['non_capped']:,.0f} "
        f"capped={results['flag']['capped']:,.0f}"
    )

    path = plot_comparison(results, outputs_dir)
    logger.info(f"\n保存しました: {path}")

    logger.info("\n=== まとめ ===")
    non_capped_delta = results["as_is"]["non_capped"] - results["drop"]["non_capped"]
    logger.info(
        f"non_capped部分のRMSEはas_is={results['as_is']['non_capped']:,.0f} → "
        f"drop={results['drop']['non_capped']:,.0f}（差={non_capped_delta:,.0f}、"
        f"{'改善した' if non_capped_delta > 0 else '悪化またはほぼ変化なし'}）。"
    )
    capped_delta = results["drop"]["capped"] - results["as_is"]["capped"]
    logger.info(
        f"capped部分のRMSEはas_is={results['as_is']['capped']:,.0f} → "
        f"drop={results['drop']['capped']:,.0f}（差={capped_delta:,.0f}）。"
        "上限行を訓練から除くと、実運用でその価格帯を予測する能力がどれだけ"
        "失われるかを示す（RandomForestは訓練データの値域を超えて外挿できない）。"
    )
    logger.info(
        f"flagパターンはcapped部分のRMSEをas_is={results['as_is']['capped']:,.0f} → "
        f"flag={results['flag']['capped']:,.0f}へ、non_capped部分のRMSEを"
        f"as_is={results['as_is']['non_capped']:,.0f} → "
        f"flag={results['flag']['non_capped']:,.0f}に保ちながら変えた。"
    )


if __name__ == "__main__":
    main()
