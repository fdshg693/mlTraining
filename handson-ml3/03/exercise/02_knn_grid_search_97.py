"""演習1: GridSearchCVでKNeighborsClassifierを調整し、テスト精度97%超を達成する。

**Warning:** 訓練データ10000件でのGridSearchCV、および60000件全体でのKNN学習・
10000件のテスト予測は、いずれも数分かかる場合がある。
"""

import json
from pathlib import Path
import sys

import numpy as np
from loguru import logger
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from data_produce.util.file_io import output_path, read_npz
from data_produce.util.logging_config import setup_logger

BEST_PARAMS_FILE = "best_knn_params.json"


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = read_npz("mnist_train.npz")
    test = read_npz("mnist_test.npz")
    return train["X"], train["y"], test["X"], test["y"]


def main() -> None:
    setup_logger(Path(__file__).stem)

    X_train, y_train, X_test, y_test = load_mnist()

    logger.info("=== ベースライン: デフォルトのKNeighborsClassifier ===")
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)
    baseline_accuracy = knn_clf.score(X_test, y_test)
    logger.info(f"ベースラインのテスト精度: {baseline_accuracy:.4f}")

    logger.info("=== GridSearchCV: weights/n_neighborsの探索(先頭10000件で高速化) ===")
    param_grid = [{"weights": ["uniform", "distance"], "n_neighbors": [3, 4, 5, 6]}]
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train[:10_000], y_train[:10_000])

    logger.info(f"best_params_: {grid_search.best_params_}")
    logger.info(
        f"best_score_(10000件・5-fold): {grid_search.best_score_:.4f} "
        "(ベースラインより下がっているが、訓練データが10000件のみのため想定通り)"
    )

    logger.info("=== 最良パラメータを訓練データ全体(60000件)で再学習 ===")
    grid_search.best_estimator_.fit(X_train, y_train)
    tuned_accuracy = grid_search.best_estimator_.score(X_test, y_test)
    logger.info(f"訓練データ全体で再学習後のテスト精度: {tuned_accuracy:.4f}")

    if tuned_accuracy > 0.97:
        logger.info("目標のテスト精度97%超を達成した")
    else:
        logger.warning("目標のテスト精度97%超を達成できなかった")

    result = {
        "best_params": grid_search.best_params_,
        "baseline_accuracy": baseline_accuracy,
        "tuned_accuracy": tuned_accuracy,
    }
    path = output_path(BEST_PARAMS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(f"{BEST_PARAMS_FILE} に最良パラメータとテスト精度を保存しました(演習2で再利用)")


if __name__ == "__main__":
    main()
