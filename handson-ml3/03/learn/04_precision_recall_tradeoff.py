"""決定関数のスコアと閾値の関係、適合率/再現率のトレードオフを確認し、
Precision/Recall vs Threshold・PR曲線を描画する。
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import cross_val_predict

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from data_produce.util.file_io import read_npz, save_figure
from data_produce.util.logging_config import setup_logger


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = read_npz("mnist_train.npz")
    test = read_npz("mnist_test.npz")
    return train["X"], train["y"], test["X"], test["y"]


def make_binary_labels(
    y_train: np.ndarray, y_test: np.ndarray, target: str = "5"
) -> tuple[np.ndarray, np.ndarray]:
    return (y_train == target), (y_test == target)


def explore_threshold_effect(sgd_clf: SGDClassifier, some_digit: np.ndarray) -> None:
    """決定関数のスコアを閾値と比較するだけでpredict()と同じ結果になることを確認する。"""

    logger.info("=== 決定関数のスコアと閾値 ===")

    y_scores = sgd_clf.decision_function([some_digit])
    logger.info(f"decision_function(some_digit)={y_scores}")

    for threshold in (0, 3000):
        y_pred = y_scores > threshold
        logger.info(f"threshold={threshold} のときの予測: {y_pred}")

    # condition.all() を使って、すべての比較が一致していることを確かめる
    logger.info(
        f"predict()と(decision_function > 0)が一致するか: "
        f"{bool((sgd_clf.predict([some_digit]) == (y_scores > 0)).all())}"
        "（=閾値0がpredict()のデフォルト挙動）"
    )
    logger.info("閾値を上げるほど「5」と判定しにくくなり、適合率は上がり再現率は下がる")


def plot_precision_recall_vs_threshold(
    precisions: np.ndarray, recalls: np.ndarray, thresholds: np.ndarray, marker_threshold: float
) -> None:
    # `argmax()`はブール配列に対しては「最大値（=True）が最初に現れるインデックス」を返す。
    idx = (thresholds >= marker_threshold).argmax()

    fig = plt.figure(figsize=(8, 4))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.vlines(marker_threshold, 0, 1.0, "k", "dotted", label="threshold")
    plt.plot(thresholds[idx], precisions[idx], "bo")
    plt.plot(thresholds[idx], recalls[idx], "go")
    plt.axis([-50000, 50000, 0, 1])
    plt.grid()
    plt.xlabel("Threshold")
    plt.legend(loc="center right")
    save_figure(fig, "precision_recall_vs_threshold_plot.png")
    plt.close(fig)


def plot_precision_vs_recall(precisions: np.ndarray, recalls: np.ndarray, idx: int) -> None:
    fig = plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
    plt.plot([recalls[idx], recalls[idx]], [0.0, precisions[idx]], "k:")
    plt.plot([0.0, recalls[idx]], [precisions[idx], precisions[idx]], "k:")
    plt.plot([recalls[idx]], [precisions[idx]], "ko", label="Point at threshold")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.axis([0, 1, 0, 1])
    plt.grid()
    plt.legend(loc="lower left")
    save_figure(fig, "precision_vs_recall_plot.png")
    plt.close(fig)


def main() -> None:
    setup_logger(Path(__file__).stem)

    X_train, y_train, X_test, y_test = load_mnist()
    y_train_5, y_test_5 = make_binary_labels(y_train, y_test)

    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train_5)
    explore_threshold_effect(sgd_clf, X_train[0])

    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    # thresholds は昇順。precisions/recalls は末尾に「閾値=無限大（何も陽性としない）」に
    # 対応する1要素（precisions[-1]=1, recalls[-1]=0）が余分に付くため、
    # thresholds と要素単位で対応させるには precisions[:-1] / recalls[:-1] を使う
    # （詳細: docs/stats.md「PR曲線とprecision_recall_curveの戻り値」）
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

    marker_threshold = 3000
    # `argmax()`はブール配列に対しては「最大値（=True）が最初に現れるインデックス」を返す。
    idx = (thresholds >= marker_threshold).argmax()
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds, marker_threshold)
    plot_precision_vs_recall(precisions, recalls, idx)
    logger.info("precision_recall_vs_threshold_plot.png, precision_vs_recall_plot.png を保存しました")

    # argmax()はブール配列中で最初にTrueになるインデックスを返す。
    # thresholdsが昇順なので「最初にTrueになる=最小の閾値」となり、
    # 「適合率90%を初めて満たす最小の閾値」が求まる（thresholdsの並び順の前提が崩れると成立しない）
    idx_for_90_precision = (precisions >= 0.90).argmax()
    threshold_for_90_precision = thresholds[idx_for_90_precision]
    logger.info(f"適合率90%を達成する最小の閾値: {threshold_for_90_precision:.2f}")

    y_train_pred_90 = y_scores >= threshold_for_90_precision
    precision_at_90 = precision_score(y_train_5, y_train_pred_90)
    recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
    logger.info(
        f"この閾値での適合率={precision_at_90:.4f}, 再現率={recall_at_90_precision:.4f}"
        "（=適合率を上げようとすると再現率が犠牲になるトレードオフ）"
    )


if __name__ == "__main__":
    main()
