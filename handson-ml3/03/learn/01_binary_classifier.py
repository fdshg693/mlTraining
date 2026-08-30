"""「5か否か」の二値分類ラベルを作り、SGDClassifierで学習・予測する。
random_state固定の意味と、SGDClassifierのオンライン学習的な性質も確認する。
"""

from pathlib import Path
import sys

import numpy as np
from loguru import logger
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

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


def explore_random_state(
    X_train: np.ndarray, y_train_5: np.ndarray, sample: np.ndarray
) -> None:
    """random_stateを固定する/しないことが学習結果の再現性に与える影響を確認する。"""

    logger.info("=== random_state固定の意味 ===")

    # decision_function()は、そのサンプルが分類境界からどれだけ離れているかを表す
    # 生のスコア（SGDClassifierの既定loss="hinge"の場合は線形SVMのマージンからの距離）を返す。
    # predict()はこのスコアが0以上ならTrue、0未満ならFalseを返すだけの薄いラッパーであり、
    # スコア自体は確率ではない（確率が欲しい場合はloss="log_loss"等にしてpredict_probaを使う）。
    # ここではモデルの「予測結果」ではなく「予測の根拠となる生の値」を比較することで、
    # 学習された重みそのものがrandom_stateによってどれだけずれるかを直接確認できる。
    clf_a = SGDClassifier(random_state=42).fit(X_train, y_train_5)
    clf_b = SGDClassifier(random_state=42).fit(X_train, y_train_5)
    same_seed_diff = np.abs(
        clf_a.decision_function(sample) - clf_b.decision_function(sample)
    ).max()
    logger.info(
        f"同じrandom_state=42で2回学習した決定関数の差(絶対値最大): {same_seed_diff:.10f}"
        "（=完全に再現される）"
    )

    clf_c = SGDClassifier(random_state=0).fit(X_train, y_train_5)
    diff_seed_diff = np.abs(
        clf_a.decision_function(sample) - clf_c.decision_function(sample)
    ).max()
    logger.info(
        f"random_state=42とrandom_state=0で学習した決定関数の差(絶対値最大): {diff_seed_diff:.4f}"
        "（=シードが違うとサンプルの処理順が変わり、重みも変わる）"
    )

    clf_d1 = SGDClassifier().fit(X_train, y_train_5)
    clf_d2 = SGDClassifier().fit(X_train, y_train_5)
    none_seed_diff = np.abs(
        clf_d1.decision_function(sample) - clf_d2.decision_function(sample)
    ).max()
    logger.info(
        f"random_state未指定で2回学習した決定関数の差(絶対値最大): {none_seed_diff:.4f}"
        "（=固定しない限り実行のたびに結果が変わりうる）"
    )


def explore_online_learning(
    X_train: np.ndarray,
    y_train_5: np.ndarray,
    X_test: np.ndarray,
    y_test_5: np.ndarray,
    batch_size: int = 1000,
) -> None:
    """fit(一括学習)とpartial_fit(ミニバッチ逐次学習)のテスト精度を比較する。"""

    logger.info("=== SGDClassifierのオンライン学習的な性質 ===")

    batch_clf = SGDClassifier(random_state=42)
    batch_clf.fit(X_train, y_train_5)
    batch_acc = accuracy_score(y_test_5, batch_clf.predict(X_test))
    logger.info(f"一括学習(fit)のテスト精度: {batch_acc:.4f}")

    online_clf = SGDClassifier(random_state=42)
    classes = np.array([False, True])
    for start in range(0, len(X_train), batch_size):
        end = start + batch_size
        online_clf.partial_fit(
            X_train[start:end], y_train_5[start:end], classes=classes
        )
    online_acc = accuracy_score(y_test_5, online_clf.predict(X_test))
    logger.info(
        f"ミニバッチ逐次学習(partial_fit, batch_size={batch_size})のテスト精度: {online_acc:.4f}"
    )
    logger.info(
        "partial_fitはデータ全体を一度にメモリへ載せず、ミニバッチ単位で重みを逐次更新できる"
        "（=大規模データやストリーミングデータに向く、オンライン学習的な性質）"
    )


def main() -> None:
    setup_logger(Path(__file__).stem)

    X_train, y_train, X_test, y_test = load_mnist()
    logger.info(f"X_train.shape={X_train.shape}, X_test.shape={X_test.shape}")

    y_train_5, y_test_5 = make_binary_labels(y_train, y_test)
    logger.info(f"訓練データ中の5の割合: {y_train_5.mean():.4f}")

    some_digit = X_train[0]
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train_5)
    prediction = sgd_clf.predict([some_digit])
    logger.info(f"y_train[0]={y_train[0]!r} に対する予測: {prediction}")

    test_acc = accuracy_score(y_test_5, sgd_clf.predict(X_test))
    logger.info(f"テスト精度: {test_acc:.4f}")

    explore_random_state(X_train, y_train_5, X_train[:5])
    explore_online_learning(X_train, y_train_5, X_test, y_test_5)


if __name__ == "__main__":
    main()
