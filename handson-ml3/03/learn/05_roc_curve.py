"""ROC曲線とAUCを算出し、RandomForestClassifier(predict_probaを利用)と
SGDClassifierの性能をPR曲線・ROC曲線・AUCで比較する。
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
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


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, mark_fpr: float, mark_tpr: float) -> None:
    fig = plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label="ROC curve (SGD)")
    plt.plot([0, 1], [0, 1], "k:", label="Random classifier's ROC curve")
    plt.plot([mark_fpr], [mark_tpr], "ko", label="Threshold for 90% precision")
    plt.xlabel("False Positive Rate (Fall-Out)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.grid()
    plt.axis([0, 1, 0, 1])
    plt.legend(loc="lower right", fontsize=10)
    save_figure(fig, "roc_curve_plot.png")
    plt.close(fig)


def plot_pr_curve_comparison(
    recalls_sgd: np.ndarray,
    precisions_sgd: np.ndarray,
    recalls_forest: np.ndarray,
    precisions_forest: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(6, 5))
    plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2, label="Random Forest")
    plt.plot(recalls_sgd, precisions_sgd, "--", linewidth=2, label="SGD")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.axis([0, 1, 0, 1])
    plt.grid()
    plt.legend(loc="lower left")
    save_figure(fig, "pr_curve_comparison_plot.png")
    plt.close(fig)


def main() -> None:
    setup_logger(Path(__file__).stem)

    X_train, y_train, X_test, y_test = load_mnist()
    y_train_5, y_test_5 = make_binary_labels(y_train, y_test)

    # SGDClassifier: 決定関数のスコアからROC曲線・AUCを求める
    sgd_clf = SGDClassifier(random_state=42)
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    fpr, tpr, roc_thresholds = roc_curve(y_train_5, y_scores)

    precisions, recalls, pr_thresholds = precision_recall_curve(y_train_5, y_scores)
    idx_for_90_precision = (precisions >= 0.90).argmax()
    threshold_for_90_precision = pr_thresholds[idx_for_90_precision]
    idx_for_threshold_at_90 = (roc_thresholds <= threshold_for_90_precision).argmax()
    tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

    plot_roc_curve(fpr, tpr, fpr_90, tpr_90)

    sgd_auc = roc_auc_score(y_train_5, y_scores)
    logger.info(f"SGDClassifierのROC AUC: {sgd_auc:.4f}")

    # RandomForestClassifier: predict_probaで陽性クラスの確率を得て同様に評価する
    logger.info("RandomForestClassifierを3-fold交差検証中(数分かかる場合があります)...")
    forest_clf = RandomForestClassifier(random_state=42)
    y_probas_forest = cross_val_predict(
        forest_clf, X_train, y_train_5, cv=3, method="predict_proba"
    )
    logger.info(f"y_probas_forest[:2]=\n{y_probas_forest[:2]}")

    idx_50_to_60 = (y_probas_forest[:, 1] > 0.50) & (y_probas_forest[:, 1] < 0.60)
    proba_50_to_60_actual_positive_rate = (y_train_5[idx_50_to_60]).sum() / idx_50_to_60.sum()
    logger.info(
        f"陽性確率50〜60%と予測された画像のうち、実際に陽性(5)だった割合: "
        f"{proba_50_to_60_actual_positive_rate:.1%}"
    )

    y_scores_forest = y_probas_forest[:, 1]
    precisions_forest, recalls_forest, _ = precision_recall_curve(y_train_5, y_scores_forest)
    plot_pr_curve_comparison(recalls, precisions, recalls_forest, precisions_forest)
    logger.info("roc_curve_plot.png, pr_curve_comparison_plot.png を保存しました")

    # cross_val_predictで得た確率をそのまま閾値50%で2値化すれば、
    # cross_val_predict(forest_clf, ..., cv=3)と同じ予測が高速に得られる
    y_train_pred_forest = y_scores_forest >= 0.5
    forest_f1 = f1_score(y_train_5, y_train_pred_forest)
    forest_auc = roc_auc_score(y_train_5, y_scores_forest)
    forest_precision = precision_score(y_train_5, y_train_pred_forest)
    forest_recall = recall_score(y_train_5, y_train_pred_forest)
    logger.info(
        f"RandomForestClassifier: F1={forest_f1:.4f}, AUC={forest_auc:.4f}, "
        f"precision={forest_precision:.4f}, recall={forest_recall:.4f}"
    )
    logger.info(
        f"SGD AUC={sgd_auc:.4f} vs RandomForest AUC={forest_auc:.4f}"
        "（=predict_probaを使えるRandomForestの方がROC AUCで上回りやすい）"
    )


if __name__ == "__main__":
    main()
