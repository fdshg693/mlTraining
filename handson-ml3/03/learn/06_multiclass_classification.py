"""SVCによるOvO方式の多クラス分類、OneVsRestClassifierによるOvR方式、
SGDClassifierの多クラス対応、StandardScalerによるスケーリング効果を確認する。
"""

from pathlib import Path
import sys

import numpy as np
from loguru import logger
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from data_produce.util.file_io import read_npz
from data_produce.util.logging_config import setup_logger


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = read_npz("mnist_train.npz")
    test = read_npz("mnist_test.npz")
    return train["X"], train["y"], test["X"], test["y"]


def explore_svc_ovo(X_train: np.ndarray, y_train: np.ndarray, some_digit: np.ndarray) -> SVC:
    """SVCによるOvO方式の多クラス分類を確認する。SVMは大規模データにスケールしにくいため
    先頭2000件のみで学習する。
    """

    logger.info("=== SVC(OvO方式)による多クラス分類 ===")
    svm_clf = SVC(random_state=42)
    svm_clf.fit(X_train[:2000], y_train[:2000])

    prediction = svm_clf.predict([some_digit])
    logger.info(f"予測: {prediction}")

    some_digit_scores = svm_clf.decision_function([some_digit])
    logger.info(
        f"decision_function(既定はovrに集約済み、{some_digit_scores.shape[1]}個): "
        f"{some_digit_scores.round(2)}"
    )

    class_id = some_digit_scores.argmax()
    logger.info(
        f"argmaxによるクラスID: {class_id} -> classes_[{class_id}]={svm_clf.classes_[class_id]}"
    )

    svm_clf.decision_function_shape = "ovo"
    some_digit_scores_ovo = svm_clf.decision_function([some_digit])
    logger.info(
        f"decision_function_shape='ovo'にすると{some_digit_scores_ovo.shape[1]}個の"
        f"ペアワイズスコアが得られる(10クラス×(10-1)/2=45): {some_digit_scores_ovo.round(2)}"
    )
    logger.info(
        "SVCは常にOvO方式で学習する(decision_function_shapeは学習方式ではなく、"
        "スコアの見せ方(集約するか45個全部出すか)だけを変える)"
    )
    return svm_clf


def explore_ovr(X_train: np.ndarray, y_train: np.ndarray, some_digit: np.ndarray) -> None:
    """OneVsRestClassifierによる明示的なOvR方式の多クラス分類を確認する。"""

    logger.info("=== OneVsRestClassifier(OvR方式)による多クラス分類 ===")
    ovr_clf = OneVsRestClassifier(SVC(random_state=42))
    ovr_clf.fit(X_train[:2000], y_train[:2000])

    prediction = ovr_clf.predict([some_digit])
    logger.info(f"予測: {prediction}, 内部の2値分類器の数: {len(ovr_clf.estimators_)}")
    logger.info(
        "OvRはクラス数と同じ数(10個)の2値分類器を、それぞれ全訓練データで学習する。"
        "OvOはクラスペアの数(45個)の分類器を、各ペアに属するデータだけで学習する。"
        "分類器1つあたりの訓練データは少なく済むが分類器の数が多いOvOと、"
        "分類器は少ないが1つあたり全データを使うOvRという計算量のトレードオフがある"
        "(一般に、多くのアルゴリズムは訓練データが増えるほど学習が遅くなるため、"
        "SVMのようなアルゴリズムではOvOが好まれる)"
    )


def explore_sgd_multiclass(
    X_train: np.ndarray, y_train: np.ndarray, some_digit: np.ndarray
) -> tuple[SGDClassifier, np.ndarray]:
    """SGDClassifierの多クラス対応(内部でOvRを使用)を確認する。"""

    logger.info("=== SGDClassifierの多クラス対応 ===")
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train)

    prediction = sgd_clf.predict([some_digit])
    logger.info(f"予測: {prediction}")

    scores = sgd_clf.decision_function([some_digit]).round()
    logger.info(f"decision_function(10クラス分のスコア): {scores}")

    logger.info("スケーリング前のデータで3-fold交差検証中(数分かかる場合があります)...")
    scores_cv = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
    logger.info(f"スケーリング前のAccuracy: {scores_cv} (平均{scores_cv.mean():.4f})")
    return sgd_clf, scores_cv


def explore_scaling_effect(
    sgd_clf: SGDClassifier, X_train: np.ndarray, y_train: np.ndarray, scores_before: np.ndarray
) -> None:
    """StandardScalerによるスケーリングがAccuracyに与える効果を確認する。"""

    logger.info("=== StandardScalerによるスケーリング効果 ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype("float64"))

    logger.info("スケーリング後のデータで3-fold交差検証中(数分かかる場合があります)...")
    scores_cv_scaled = cross_val_score(
        sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"
    )
    logger.info(f"スケーリング後のAccuracy: {scores_cv_scaled} (平均{scores_cv_scaled.mean():.4f})")
    logger.info(
        f"平均Accuracyの変化: {scores_before.mean():.4f} -> {scores_cv_scaled.mean():.4f}"
        "(=画素値0〜255を平均0・分散1にスケーリングすると、勾配降下ベースのSGDが"
        "各特徴量を均等に扱えるようになり収束しやすくなる)"
    )


def main() -> None:
    setup_logger(Path(__file__).stem)

    X_train, y_train, _X_test, _y_test = load_mnist()
    some_digit = X_train[0]
    logger.info(f"X_train.shape={X_train.shape}, y_train[0]={y_train[0]!r}")

    explore_svc_ovo(X_train, y_train, some_digit)
    explore_ovr(X_train, y_train, some_digit)
    sgd_clf, scores_cv = explore_sgd_multiclass(X_train, y_train, some_digit)
    explore_scaling_effect(sgd_clf, X_train, y_train, scores_cv)


if __name__ == "__main__":
    main()
