"""演習2: shift()によるデータ拡張で訓練データを5倍(60000→300000件)に増やし、
演習1(exercise/02)で見つけた最良ハイパーパラメータのKNeighborsClassifierを
再学習してテスト精度・エラー率の変化を確認する。

**Warning:** 300000件での学習・10000件のテスト予測は、演習1よりもさらに時間がかかる
(数分〜十数分程度)。
"""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.ndimage import shift
from sklearn.neighbors import KNeighborsClassifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from data_produce.util.file_io import output_path, read_npz, save_figure
from data_produce.util.logging_config import setup_logger

BEST_PARAMS_FILE = "best_knn_params.json"
DEFAULT_BEST_PARAMS = {"n_neighbors": 4, "weights": "distance"}


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = read_npz("mnist_train.npz")
    test = read_npz("mnist_test.npz")
    return train["X"], train["y"], test["X"], test["y"]


def load_previous_result() -> dict:
    """演習1(exercise/02)の出力から最良パラメータ・拡張前のテスト精度を読み込む。
    存在しない場合は原本ノートブックと同じデフォルト値にフォールバックする。
    """

    path = output_path(BEST_PARAMS_FILE, outputs_dir=PROJECT_DIR / "exercise" / "outputs")
    if path.is_file():
        result = json.loads(path.read_text(encoding="utf-8"))
        logger.info(f"演習1の結果を読み込みました: {result}")
        return result
    logger.warning(
        f"{path} が見つからないため、デフォルトのbest_params {DEFAULT_BEST_PARAMS} を使用します"
        "(先にexercise/02_knn_grid_search_97.pyを実行すると演習1の結果を再利用できます)"
    )
    return {"best_params": DEFAULT_BEST_PARAMS, "tuned_accuracy": None}


def shift_image(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])


def demo_shift(X_train: np.ndarray) -> None:
    image = X_train[1000]
    shifted_down = shift_image(image, 0, 5)
    shifted_left = shift_image(image, -5, 0)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    for ax, image_data, title in [
        (axs[0], image, "Original"),
        (axs[1], shifted_down, "Shifted down"),
        (axs[2], shifted_left, "Shifted left"),
    ]:
        ax.imshow(image_data.reshape(28, 28), cmap="Greys", interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    save_figure(fig, "shift_demo_plot.png")
    plt.close(fig)
    logger.info("shift_demo_plot.png を保存しました")


def augment_dataset(
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """上下左右に1画素ずつシフトした画像を追加し、訓練データを5倍(60000→300000件)にする。"""

    X_augmented = [image for image in X_train]
    y_augmented = [label for label in y_train]

    for dx, dy in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        for image, label in zip(X_train, y_train):
            X_augmented.append(shift_image(image, dx, dy))
            y_augmented.append(label)

    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)

    rng = np.random.default_rng(42)
    shuffle_idx = rng.permutation(len(X_augmented))
    return X_augmented[shuffle_idx], y_augmented[shuffle_idx]


def main() -> None:
    setup_logger(Path(__file__).stem)

    X_train, y_train, X_test, y_test = load_mnist()
    previous_result = load_previous_result()
    best_params = previous_result["best_params"]
    tuned_accuracy = previous_result.get("tuned_accuracy")

    demo_shift(X_train)

    logger.info("=== データ拡張: 上下左右に1画素シフトした画像を追加 ===")
    X_train_augmented, y_train_augmented = augment_dataset(X_train, y_train)
    logger.info(
        f"拡張前: {X_train.shape[0]}件 -> 拡張後: {X_train_augmented.shape[0]}件"
        f"({X_train_augmented.shape[0] // X_train.shape[0]}倍)"
    )

    logger.info(f"=== 演習1の最良パラメータ{best_params}で拡張データを学習 ===")
    knn_clf = KNeighborsClassifier(**best_params)
    knn_clf.fit(X_train_augmented, y_train_augmented)

    augmented_accuracy = knn_clf.score(X_test, y_test)
    logger.info(f"データ拡張後のテスト精度: {augmented_accuracy:.4f}")

    if tuned_accuracy is not None:
        logger.info(f"データ拡張前(演習1)のテスト精度: {tuned_accuracy:.4f}")
        accuracy_gain = augmented_accuracy - tuned_accuracy
        error_rate_change = (1 - augmented_accuracy) / (1 - tuned_accuracy) - 1
        logger.info(
            f"精度の変化: {accuracy_gain:+.4f}, エラー率の変化: {error_rate_change:+.0%} "
            "(精度の伸びは小さく見えても、エラー率で見ると下がり幅が大きいことが多い)"
        )
    else:
        logger.info(
            "演習1の拡張前テスト精度が無いため、エラー率変化の比較は省略(演習1を先に実行すると比較可能)"
        )


if __name__ == "__main__":
    main()
