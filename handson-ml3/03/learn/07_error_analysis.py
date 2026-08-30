"""混同行列の正規化・可視化と、誤分類しやすい数字ペア（3と5）の画像確認を行う。

**Warning:** cross_val_predictでの3-fold交差検証は数分かかる。
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from data_produce.util.file_io import read_npz, save_figure
from data_produce.util.logging_config import setup_logger


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = read_npz("mnist_train.npz")
    test = read_npz("mnist_test.npz")
    return train["X"], train["y"], test["X"], test["y"]


def get_cross_val_predictions(X_train_scaled: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """StandardScaler適用後のデータに対する3-fold交差検証での予測値を返す。"""

    sgd_clf = SGDClassifier(random_state=42)
    logger.info("SGDClassifierを3-fold交差検証中(数分かかる場合があります)...")
    return cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)


def plot_confusion_matrix_pair(y_train: np.ndarray, y_train_pred: np.ndarray) -> None:
    """通常の混同行列と、行ごとに正規化した混同行列を並べて描く。"""

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    plt.rc("font", size=9)
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=axs[0])
    axs[0].set_title("Confusion matrix")
    plt.rc("font", size=10)
    ConfusionMatrixDisplay.from_predictions(
        y_train, y_train_pred, ax=axs[1], normalize="true", values_format=".0%"
    )
    axs[1].set_title("CM normalized by row")
    save_figure(fig, "confusion_matrix_plot.png")
    plt.close(fig)


def plot_error_normalized_pair(y_train: np.ndarray, y_train_pred: np.ndarray) -> None:
    """誤分類だけに絞り、行(真のクラス)・列(予測クラス)それぞれで正規化した
    混同行列を並べて描く。対角成分(正解)の重みを0にすることで、
    誤分類パターンだけが浮かび上がる。
    """

    sample_weight = y_train_pred != y_train
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    plt.rc("font", size=10)
    ConfusionMatrixDisplay.from_predictions(
        y_train,
        y_train_pred,
        ax=axs[0],
        sample_weight=sample_weight,
        normalize="true",
        values_format=".0%",
    )
    axs[0].set_title("Errors normalized by row")
    ConfusionMatrixDisplay.from_predictions(
        y_train,
        y_train_pred,
        ax=axs[1],
        sample_weight=sample_weight,
        normalize="pred",
        values_format=".0%",
    )
    axs[1].set_title("Errors normalized by column")
    save_figure(fig, "confusion_matrix_errors_plot.png")
    plt.close(fig)
    plt.rc("font", size=10)


def plot_digit_pair_grid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    cl_a: str,
    cl_b: str,
    file_name: str,
) -> None:
    """2クラス(cl_a, cl_b)について、正解/誤分類の組み合わせ4パターンを
    5x5の画像グリッドとして並べる(左上=b->a誤分類, 右下=a->b誤分類)。
    """

    X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
    X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
    X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
    X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

    size = 5
    pad = 0.2
    fig = plt.figure(figsize=(size, size))
    for images, (label_col, label_row) in [
        (X_ba, (0, 0)),
        (X_bb, (1, 0)),
        (X_aa, (0, 1)),
        (X_ab, (1, 1)),
    ]:
        for idx, image_data in enumerate(images[: size * size]):
            x = idx % size + label_col * (size + pad)
            y = idx // size + label_row * (size + pad)
            plt.imshow(
                image_data.reshape(28, 28), cmap="binary", extent=(x, x + 1, y, y + 1)
            )
    plt.xticks([size / 2, size + pad + size / 2], [str(cl_a), str(cl_b)])
    plt.yticks([size / 2, size + pad + size / 2], [str(cl_b), str(cl_a)])
    plt.plot([size + pad / 2, size + pad / 2], [0, 2 * size + pad], "k:")
    plt.plot([0, 2 * size + pad], [size + pad / 2, size + pad / 2], "k:")
    plt.axis([0, 2 * size + pad, 0, 2 * size + pad])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    save_figure(fig, file_name)
    plt.close(fig)

    logger.info(
        f"'{cl_a}'->'{cl_b}'誤分類: {len(X_ab)}件, "
        f"'{cl_b}'->'{cl_a}'誤分類: {len(X_ba)}件 "
        f"('{cl_a}'正解: {len(X_aa)}件, '{cl_b}'正解: {len(X_bb)}件)"
    )


def main() -> None:
    setup_logger(Path(__file__).stem)

    X_train, y_train, _X_test, _y_test = load_mnist()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype("float64"))

    y_train_pred = get_cross_val_predictions(X_train_scaled, y_train)

    cm = confusion_matrix(y_train, y_train_pred)
    logger.info(f"混同行列(10x10)の対角成分(正解数): {np.diag(cm)}")

    plot_confusion_matrix_pair(y_train, y_train_pred)
    logger.info(
        "confusion_matrix_plot.png を保存しました"
        "(通常の混同行列は件数の多いクラスが目立ちやすく、パッと見での比較が難しい"
        "→行ごとに正規化(normalize='true')すると、各真のクラス内での予測先の"
        "割合として比較できるようになる)"
    )

    plot_error_normalized_pair(y_train, y_train_pred)
    logger.info(
        "confusion_matrix_errors_plot.png を保存しました"
        "(sample_weightで正解(対角成分)の重みを0にし、誤分類だけに絞って正規化する"
        "ことで、どの数字がどの数字に間違われやすいかがより明確になる)"
    )

    plot_digit_pair_grid(
        X_train, y_train, y_train_pred, "3", "5", "error_analysis_3_5_plot.png"
    )
    logger.info(
        "error_analysis_3_5_plot.png を保存しました"
        "(3と5は曲線の丸みが似ており、筆跡によっては人間の目にも紛らわしい"
        "→エラー分析は単に精度を見るだけでなく、モデルがどこで人間的にも"
        "納得できる間違え方をしているかを確認する手段になる)"
    )


if __name__ == "__main__":
    main()
