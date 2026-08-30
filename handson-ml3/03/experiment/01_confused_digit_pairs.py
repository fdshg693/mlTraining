"""learn/07で行った3と5の誤分類ペア分析を他の数字ペアにも広げ、
どの数字の組み合わせが最も混同されやすいかを調べる実験。

**Warning:** cross_val_predictでの3-fold交差検証は数分かかる。

【実行前に予想】3と5以外にも、形が似た数字ペア(4と9、7と9、
偶奇はともかく丸みのある3・8・5・6など)は混同されやすいはず。
特に4と9は縦棒+丸という共通構造を持つため、上位に来るのではないか。
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from data_produce.util.file_io import read_npz, save_figure  # noqa: E402
from data_produce.util.logging_config import setup_logger  # noqa: E402

TOP_N_PAIRS = 5


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = read_npz("mnist_train.npz")
    test = read_npz("mnist_test.npz")
    return train["X"], train["y"], test["X"], test["y"]


def get_cross_val_predictions(X_train_scaled: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    sgd_clf = SGDClassifier(random_state=42)
    logger.info("SGDClassifierを3-fold交差検証中(数分かかる場合があります)...")
    return cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)


def rank_confused_pairs(
    y_train: np.ndarray, y_train_pred: np.ndarray, classes: np.ndarray
) -> list[tuple[str, str, int]]:
    """クラスペア(a, b)ごとに a<->b の誤分類件数の合計を求め、多い順に並べる。"""

    cm = confusion_matrix(y_train, y_train_pred, labels=classes)
    pairs: list[tuple[str, str, int]] = []
    for i, cl_a in enumerate(classes):
        for j, cl_b in enumerate(classes):
            if j <= i:
                continue
            confusion_count = int(cm[i, j] + cm[j, i])
            pairs.append((cl_a, cl_b, confusion_count))
    pairs.sort(key=lambda pair: pair[2], reverse=True)
    return pairs


def plot_digit_pair_grid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    cl_a: str,
    cl_b: str,
    file_name: str,
) -> None:
    """learn/07_error_analysis.pyと同じ4象限(正解/誤分類x2クラス)の画像グリッド。"""

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


def plot_pair_ranking(pairs: list[tuple[str, str, int]], top_n: int) -> None:
    """誤分類件数の多いペアTOP_N_PAIRSを横棒グラフで示す。"""

    top_pairs = pairs[:top_n]
    labels = [f"{a}<->{b}" for a, b, _ in top_pairs]
    counts = [count for _, _, count in top_pairs]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(labels[::-1], counts[::-1])
    ax.set_xlabel("Confused count (a->b + b->a)")
    ax.set_title(f"Top {top_n} most confused digit pairs")
    fig.tight_layout()
    save_figure(fig, "confused_pairs_ranking_plot.png")
    plt.close(fig)


def main() -> None:
    setup_logger(Path(__file__).stem)

    X_train, y_train, _X_test, _y_test = load_mnist()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype("float64"))

    y_train_pred = get_cross_val_predictions(X_train_scaled, y_train)

    classes = np.unique(y_train)
    pairs = rank_confused_pairs(y_train, y_train_pred, classes)

    logger.info(f"=== 誤分類件数が多い数字ペア TOP{TOP_N_PAIRS} ===")
    for rank, (cl_a, cl_b, count) in enumerate(pairs[:TOP_N_PAIRS], start=1):
        logger.info(f"{rank}位: '{cl_a}' <-> '{cl_b}' ({count}件)")

    plot_pair_ranking(pairs, TOP_N_PAIRS)
    logger.info("confused_pairs_ranking_plot.png を保存しました")

    for cl_a, cl_b, count in pairs[:TOP_N_PAIRS]:
        file_name = f"error_analysis_{cl_a}_{cl_b}_plot.png"
        plot_digit_pair_grid(X_train, y_train, y_train_pred, cl_a, cl_b, file_name)
        logger.info(f"{file_name} を保存しました('{cl_a}'<->'{cl_b}', {count}件)")

    top_a, top_b, top_count = pairs[0]
    logger.info(
        f"最も混同されやすいペアは '{top_a}' <-> '{top_b}' ({top_count}件)。"
        "learn/07_error_analysis.pyで確認した'3'<->'5'のペアと比較し、"
        "どの数字が形の近さゆえに混同されているかを画像で確認できる"
    )


if __name__ == "__main__":
    main()
