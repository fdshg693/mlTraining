"""KNeighborsClassifierとClassifierChainによる多ラベル分類(大きい数字/奇数)、
KNeighborsClassifierによる多出力分類(ノイズ除去、画像→画像の回帰的分類)を確認する。

**Warning:** KNeighborsClassifierでの3-fold交差検証は数分かかる。
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.multioutput import ClassifierChain
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from data_produce.util.file_io import read_npz, save_figure
from data_produce.util.logging_config import setup_logger


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = read_npz("mnist_train.npz")
    test = read_npz("mnist_test.npz")
    return train["X"], train["y"], test["X"], test["y"]


def make_multilabel_targets(y_train: np.ndarray) -> np.ndarray:
    """「7以上か」「奇数か」の2つの二値ラベルを列として持つ多ラベル配列を作る。"""

    y_train_large = y_train >= "7"
    y_train_odd = y_train.astype("int8") % 2 == 1
    return np.c_[y_train_large, y_train_odd]


def explore_knn_multilabel(
    X_train: np.ndarray, y_multilabel: np.ndarray, some_digit: np.ndarray
) -> None:
    """KNeighborsClassifierによる多ラベル分類と、交差検証によるF1スコア評価を確認する。"""

    logger.info("=== KNeighborsClassifierによる多ラベル分類 ===")
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_multilabel)

    prediction = knn_clf.predict([some_digit])
    logger.info(
        f"予測(7以上か, 奇数か): {prediction} "
        "(KNeighborsClassifierは複数列のyをそのまま渡すだけで多ラベル分類に対応する)"
    )

    logger.info("3-fold交差検証で予測中(数分かかる場合があります)...")
    y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)

    f1_macro = f1_score(y_multilabel, y_train_knn_pred, average="macro")
    f1_weighted = f1_score(y_multilabel, y_train_knn_pred, average="weighted")
    logger.info(
        f"F1スコア(average='macro'): {f1_macro:.4f}, "
        f"F1スコア(average='weighted'): {f1_weighted:.4f} "
        "(2つのラベルの件数比がほぼ均等なため、クラス件数で重み付けするweightedと"
        "macroとの差はごくわずかしかない)"
    )


def explore_classifier_chain(
    X_train: np.ndarray, y_multilabel: np.ndarray, some_digit: np.ndarray
) -> None:
    """ClassifierChainによる多ラベル分類を確認する。SVMは大規模データにスケールしにくいため
    先頭2000件のみで学習する。
    """

    logger.info("=== ClassifierChainによる多ラベル分類 ===")
    chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
    chain_clf.fit(X_train[:2000], y_multilabel[:2000])

    prediction = chain_clf.predict([some_digit])
    logger.info(
        f"予測(7以上か, 奇数か): {prediction} "
        "(ClassifierChainは各ラベルを2値分類器の連鎖で予測し、後段の分類器は前段の"
        "予測ラベルも特徴量として利用できる点がKNeighborsClassifierとの違い)"
    )


def explore_noise_removal(X_train: np.ndarray, X_test: np.ndarray) -> None:
    """多出力分類によるノイズ除去(画像→画像の回帰的分類)を確認する。
    各画素値(0〜255の256クラス)を予測する多出力多クラス分類として扱う。
    """

    logger.info("=== 多出力分類によるノイズ除去 ===")
    rng = np.random.default_rng(42)
    noise_train = rng.integers(0, 100, (len(X_train), 784))
    X_train_mod = X_train + noise_train
    noise_test = rng.integers(0, 100, (len(X_test), 784))
    X_test_mod = X_test + noise_test
    y_train_mod = X_train
    y_test_mod = X_test

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train_mod, y_train_mod)
    clean_digit = knn_clf.predict([X_test_mod[0]])

    error = np.abs(clean_digit[0].astype("int16") - y_test_mod[0].astype("int16"))
    logger.info(
        f"復元画像と元画像の画素値の平均絶対誤差: {error.mean():.2f} "
        "(入力はノイズを加えた画像、出力/正解ラベルはノイズなしの元画像の各画素値。"
        "784画素それぞれを0〜255の多クラス分類として同時に予測するのが多出力分類)"
    )

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    for ax, image_data, title in [
        (axs[0], X_test_mod[0], "Noisy input"),
        (axs[1], clean_digit[0], "KNN prediction"),
        (axs[2], y_test_mod[0], "Ground truth"),
    ]:
        ax.imshow(image_data.reshape(28, 28), cmap="binary")
        ax.set_title(title)
        ax.axis("off")
    save_figure(fig, "noise_removal_plot.png")
    plt.close(fig)
    logger.info("noise_removal_plot.png を保存しました")


def main() -> None:
    setup_logger(Path(__file__).stem)

    X_train, y_train, X_test, _y_test = load_mnist()
    some_digit = X_train[0]
    logger.info(f"X_train.shape={X_train.shape}, y_train[0]={y_train[0]!r}")

    y_multilabel = make_multilabel_targets(y_train)
    explore_knn_multilabel(X_train, y_multilabel, some_digit)
    explore_classifier_chain(X_train, y_multilabel, some_digit)

    explore_noise_removal(X_train, X_test)


if __name__ == "__main__":
    main()
