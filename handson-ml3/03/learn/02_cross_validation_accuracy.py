"""交差検証によるAccuracy測定と、DummyClassifierとの比較を行う。
Accuracyだけでは不十分な理由を、クラス比が偏ったデータで体感する。
"""

from pathlib import Path
import sys

import numpy as np
from loguru import logger
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

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


def manual_cross_validation(
    X_train: np.ndarray, y_train_5: np.ndarray, sgd_clf: SGDClassifier
) -> list[float]:
    """cross_val_scoreの中身をStratifiedKFoldで手動再現し、同じ結果になるか確認する。"""

    logger.info("=== StratifiedKFoldによる手動再現 ===")

    skfolds = StratifiedKFold(n_splits=3)
    scores = []
    for train_index, test_index in skfolds.split(X_train, y_train_5):
        clone_clf = clone(sgd_clf)
        X_train_folds = X_train[train_index]
        y_train_folds = y_train_5[train_index]
        X_test_fold = X_train[test_index]
        y_test_fold = y_train_5[test_index]

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = int((y_pred == y_test_fold).sum())
        fold_acc = n_correct / len(y_pred)
        scores.append(fold_acc)
        logger.info(f"fold accuracy: {fold_acc:.4f}")

    return scores


def compare_with_dummy(X_train: np.ndarray, y_train_5: np.ndarray) -> None:
    """DummyClassifier（常に多数派クラスを予測）とAccuracyを比較する。"""

    logger.info("=== DummyClassifierとの比較 ===")

    # strategyを省略するとデフォルトは"prior"になり、
    # 訓練データで最も多いクラス（ここでは「5でない」）を常に予測する
    # （predict()の挙動は"most_frequent"と同じ。"prior"はpredict_probaが
    # 実際のクラス比を返す点のみ"most_frequent"と異なる）
    dummy_clf = DummyClassifier()
    dummy_clf.fit(X_train, y_train_5)
    predicts_any_5 = bool(any(dummy_clf.predict(X_train)))
    logger.info(f"DummyClassifierが1回でも5と予測するか: {predicts_any_5}（=常にFalseを予測）")

    dummy_scores = cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy")
    logger.info(f"DummyClassifierの交差検証Accuracy: {dummy_scores}")
    logger.info(
        f"訓練データ中で5でない割合: {(~y_train_5).mean():.4f}"
        "（=DummyClassifierのAccuracyとほぼ一致する）"
    )
    logger.info(
        "常に多数派クラス(5でない)を予測するだけでAccuracy約90%が出てしまうため、"
        "Accuracy単体では「5を正しく見つけられているか」を評価できない"
        "（=クラス比が偏ったデータでAccuracyが不十分な理由）"
    )


def main() -> None:
    setup_logger(Path(__file__).stem)

    X_train, y_train, X_test, y_test = load_mnist()
    y_train_5, y_test_5 = make_binary_labels(y_train, y_test)

    sgd_clf = SGDClassifier(random_state=42)
    cv_scores = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
    logger.info(f"SGDClassifierの交差検証Accuracy(cross_val_score): {cv_scores}")

    manual_scores = manual_cross_validation(X_train, y_train_5, sgd_clf)
    logger.info(
        f"手動再現との差(絶対値最大): {np.abs(cv_scores - np.array(manual_scores)).max():.10f}"
        "（=cross_val_scoreは内部でStratifiedKFoldと同等の分割・評価を行っている）"
    )

    compare_with_dummy(X_train, y_train_5)


if __name__ == "__main__":
    main()
