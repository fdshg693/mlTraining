"""混同行列・適合率(Precision)・再現率(Recall)・F1スコアを算出し、
混同行列からの手計算でsklearnの結果を検算する。
"""

from pathlib import Path
import sys

import numpy as np
from loguru import logger
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
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


def verify_against_confusion_matrix(cm: np.ndarray, precision: float, recall: float, f1: float) -> None:
    """混同行列の各セルから、precision/recall/f1をsklearnを使わず手計算して検算する。"""

    logger.info("=== 混同行列からの手計算による検算 ===")

    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    logger.info(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    manual_precision = tp / (fp + tp)
    manual_recall = tp / (fn + tp)
    manual_f1 = tp / (tp + (fn + fp) / 2)

    logger.info(
        f"precision_score={precision:.6f} / 手計算 TP/(FP+TP)={manual_precision:.6f} "
        f"(差={abs(precision - manual_precision):.10f})"
    )
    logger.info(
        f"recall_score={recall:.6f} / 手計算 TP/(FN+TP)={manual_recall:.6f} "
        f"(差={abs(recall - manual_recall):.10f})"
    )
    logger.info(
        f"f1_score={f1:.6f} / 手計算 TP/(TP+(FN+FP)/2)={manual_f1:.6f} "
        f"(差={abs(f1 - manual_f1):.10f})"
    )


def main() -> None:
    setup_logger(Path(__file__).stem)

    X_train, y_train, X_test, y_test = load_mnist()
    y_train_5, y_test_5 = make_binary_labels(y_train, y_test)

    sgd_clf = SGDClassifier(random_state=42)
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

    cm = confusion_matrix(y_train_5, y_train_pred)
    logger.info(f"混同行列(行=正解, 列=予測):\n{cm}")

    cm_perfect = confusion_matrix(y_train_5, y_train_5)
    logger.info(f"完全予測だった場合の混同行列(対角成分以外は0になるはず):\n{cm_perfect}")

    precision = precision_score(y_train_5, y_train_pred)
    recall = recall_score(y_train_5, y_train_pred)
    f1 = f1_score(y_train_5, y_train_pred)
    logger.info(f"precision_score={precision:.4f}, recall_score={recall:.4f}, f1_score={f1:.4f}")

    verify_against_confusion_matrix(cm, precision, recall, f1)


if __name__ == "__main__":
    main()
