"""learn/04では「適合率90%」を満たす閾値を求めた。
ここでは逆に「目標再現率90%」を満たす閾値を求め、そのときの適合率を計算する。
"""

from pathlib import Path
import sys

import numpy as np
from loguru import logger
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import cross_val_predict

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from data_produce.util.file_io import read_npz
from data_produce.util.logging_config import setup_logger


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = read_npz("mnist_train.npz")
    test = read_npz("mnist_test.npz")
    return train["X"], train["y"], test["X"], test["y"]


def make_binary_labels(
    y_train: np.ndarray, y_test: np.ndarray, target: str = "5"
) -> tuple[np.ndarray, np.ndarray]:
    return (y_train == target), (y_test == target)


def find_threshold_for_recall(
    recalls: np.ndarray, thresholds: np.ndarray, target_recall: float
) -> float:
    """再現率(recalls)は閾値が上がるほど単調非増加なので、
    target_recall以上を満たす中で最大の閾値を探す。
    """

    # recalls[-1]は閾値=+infに対応する末尾要素で、thresholdsには対応する値がないため除く
    valid = recalls[:-1] >= target_recall
    idx = int(valid.sum()) - 1
    idx = min(idx, len(thresholds) - 1)
    return thresholds[idx]


def main() -> None:
    setup_logger(Path(__file__).stem)

    X_train, y_train, X_test, y_test = load_mnist()
    y_train_5, y_test_5 = make_binary_labels(y_train, y_test)

    sgd_clf = SGDClassifier(random_state=42)
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

    target_recall = 0.90
    threshold_for_90_recall = find_threshold_for_recall(recalls, thresholds, target_recall)
    logger.info(f"目標再現率{target_recall:.0%}を達成する最大の閾値: {threshold_for_90_recall:.2f}")

    y_train_pred_90recall = y_scores >= threshold_for_90_recall
    actual_recall = recall_score(y_train_5, y_train_pred_90recall)
    precision_at_90_recall = precision_score(y_train_5, y_train_pred_90recall)
    logger.info(
        f"この閾値での再現率={actual_recall:.4f}(目標{target_recall:.0%}以上), "
        f"適合率={precision_at_90_recall:.4f}"
    )
    logger.info(
        "learn/04で求めた「適合率90%のときの再現率」と比べ、"
        "「再現率90%のときの適合率」は同じトレードオフ曲線上の別の点になる"
    )


if __name__ == "__main__":
    main()
