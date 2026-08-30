# 原本のノートブックはfetch_openml('mnist_784')でMNISTを取得しているが、
# 実行時点でOpenMLのAPI（api.openml.org / www.openml.org）が504 Gateway Timeoutを
# 返し続け取得できなかったため、取得元をtensorflow.keras.datasets.mnistが内部で
# 使っている公開npz（storage.googleapis.com）に変更した。tensorflow自体は重い依存
# なので、urlretrieveでnpzファイルを直接取得するだけに留めている。
#
# この配布元は元々(28, 28)のuint8画像・uint8ラベルを訓練60000件/テスト10000件に
# 分割済みで提供している。原本（fetch_openml版）と扱いを揃えるため、画像は
# (N, 784)に平坦化し、ラベルは文字列（'0'〜'9'）に変換してdata/へ保存する。

from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np

from data_produce.util.file_io import data_path, exists, read_npz, save_figure, write_npz
from data_produce.util.logging_config import setup_logger

logger = setup_logger(Path(__file__).stem)

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
RAW_FILE = "mnist_raw.npz"
TRAIN_FILE = "mnist_train.npz"
TEST_FILE = "mnist_test.npz"


def fetch_raw():
    if exists(RAW_FILE):
        logger.info(f"{RAW_FILE} が既に存在するため、ダウンロードをスキップします")
    else:
        logger.info(f"{MNIST_URL} からMNISTデータを取得します")
        path = data_path(RAW_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(MNIST_URL, path)

    raw = read_npz(RAW_FILE)
    X_train = raw["x_train"].reshape(-1, 784)
    X_test = raw["x_test"].reshape(-1, 784)
    y_train = raw["y_train"].astype(str)
    y_test = raw["y_test"].astype(str)
    return X_train, X_test, y_train, y_test


def plot_digit(image_data) -> None:
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")


def main() -> None:
    X_train, X_test, y_train, y_test = fetch_raw()

    # データ構造の確認: 70000枚 × 784画素（28×28）のグレースケール画像、ラベルは文字列の数字
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    logger.info(f"X.shape={X.shape}, X.dtype={X.dtype}")
    logger.info(f"y.shape={y.shape}, y.dtype={y.dtype}")
    assert X.shape == (70000, 784)
    assert y.shape == (70000,)
    logger.info(f"先頭サンプルのラベル: y[0]={y[0]!r}")

    # 可視化によるサニティチェック: 先頭画像とラベルが一致しているか目視確認する
    fig = plt.figure()
    plot_digit(X[0])
    save_figure(fig, "some_digit_plot.png")
    plt.close(fig)

    fig = plt.figure(figsize=(9, 9))
    for idx, image_data in enumerate(X[:100]):
        plt.subplot(10, 10, idx + 1)
        plot_digit(image_data)
    plt.subplots_adjust(wspace=0, hspace=0)
    save_figure(fig, "more_digits_plot.png")
    plt.close(fig)

    # 訓練/テスト分割: 配布元が既に先頭60000件を訓練用・残り10000件をテスト用として
    # 分割済み（原本ノートブックの「先頭60000件が訓練用」という前提と同じ分割）
    logger.info(f"train: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"test:  X={X_test.shape}, y={y_test.shape}")

    write_npz({"X": X_train, "y": y_train}, TRAIN_FILE)
    write_npz({"X": X_test, "y": y_test}, TEST_FILE)
    logger.info(f"{TRAIN_FILE} と {TEST_FILE} をdata/に保存しました")


if __name__ == "__main__":
    main()
