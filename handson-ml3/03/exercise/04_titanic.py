"""演習3: タイタニックデータセットで生存予測に取り組む。

数値/カテゴリ変数を分けて前処理するパイプライン(ColumnTransformer)を組み、
RandomForestClassifierとSVCを交差検証で比較する。さらに特徴量エンジニアリング
(年齢層・同乗家族数)の効果も確認する。
"""

from pathlib import Path
import sys
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from data_produce.util.logging_config import setup_logger

TITANIC_URL = "https://github.com/ageron/data/raw/main/titanic.tgz"
TITANIC_DIR = Path(__file__).resolve().parent / "data" / "titanic"

NUM_ATTRIBS = ["Age", "SibSp", "Parch", "Fare"]
CAT_ATTRIBS = ["Pclass", "Sex", "Embarked"]


def fetch_titanic_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    tarball_path = TITANIC_DIR.parent / "titanic.tgz"
    if not TITANIC_DIR.is_dir():
        tarball_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"{TITANIC_URL} からタイタニックデータを取得します")
        urllib.request.urlretrieve(TITANIC_URL, tarball_path)
        with tarfile.open(tarball_path) as titanic_tarball:
            titanic_tarball.extractall(path=TITANIC_DIR.parent, filter="data")
    else:
        logger.info(f"{TITANIC_DIR} が既に存在するため、ダウンロードをスキップします")

    train_data = pd.read_csv(TITANIC_DIR / "train.csv")
    test_data = pd.read_csv(TITANIC_DIR / "test.csv")
    return train_data, test_data


def build_preprocess_pipeline() -> ColumnTransformer:
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("ordinal_encoder", OrdinalEncoder()),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("cat_encoder", OneHotEncoder(sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, NUM_ATTRIBS),
        ("cat", cat_pipeline, CAT_ATTRIBS),
    ])


def explore_data(train_data: pd.DataFrame) -> None:
    logger.info("=== データ探索 ===")
    n_rows, n_cols = train_data.shape
    logger.info(f"train_data.shape={train_data.shape}")

    null_counts = train_data[["Age", "Cabin", "Embarked"]].isna().sum()
    logger.info(
        f"欠損値の件数(全{n_rows}件中): Age={null_counts['Age']}, "
        f"Cabin={null_counts['Cabin']}({null_counts['Cabin'] / n_rows:.0%}), "
        f"Embarked={null_counts['Embarked']}"
    )

    survival_rate = train_data["Survived"].mean()
    logger.info(f"生存率: {survival_rate:.1%}(4割弱なので、Accuracyでもある程度評価に使える)")


def compare_classifiers(X_train: np.ndarray, y_train: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    logger.info("=== RandomForestClassifier vs SVC(10-fold交差検証) ===")
    forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
    logger.info(f"RandomForest: mean={forest_scores.mean():.4f}, std={forest_scores.std():.4f}")

    svm_clf = SVC(gamma="auto")
    svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
    logger.info(f"SVC: mean={svm_scores.mean():.4f}, std={svm_scores.std():.4f}")

    if svm_scores.mean() > forest_scores.mean():
        logger.info("SVCの方が平均スコアが高く、スプレッドの傾向次第では汎化しやすい可能性がある")
    else:
        logger.info("RandomForestの方が平均スコアが高い")

    return forest_scores, svm_scores


def feature_engineering_glance(train_data: pd.DataFrame) -> None:
    logger.info("=== 特徴量エンジニアリングの手がかり ===")
    train_data = train_data.copy()
    train_data["AgeBucket"] = train_data["Age"] // 15 * 15
    age_bucket_survival = train_data[["AgeBucket", "Survived"]].groupby(["AgeBucket"]).mean()
    logger.info(f"年齢層ごとの生存率:\n{age_bucket_survival}")

    train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]
    relatives_survival = (
        train_data[["RelativesOnboard", "Survived"]].groupby(["RelativesOnboard"]).mean()
    )
    logger.info(f"同乗家族数ごとの生存率:\n{relatives_survival}")
    logger.info(
        "年齢層・同乗家族数はSurvivedとの相関が見られ、SibSp/Parchをそのまま使うより"
        "有用な特徴量になりうる(本演習では比較のみ行い、パイプラインへの組み込みは行わない)"
    )


def main() -> None:
    setup_logger(Path(__file__).stem)

    train_data, test_data = fetch_titanic_data()
    train_data = train_data.set_index("PassengerId")
    test_data = test_data.set_index("PassengerId")

    explore_data(train_data)

    preprocess_pipeline = build_preprocess_pipeline()
    X_train = preprocess_pipeline.fit_transform(train_data)
    y_train = train_data["Survived"]
    logger.info(f"前処理後の特徴量: X_train.shape={X_train.shape}")

    forest_scores, svm_scores = compare_classifiers(X_train, y_train)

    logger.info("=== テストデータへの予測(参考: Kaggleにはラベルがないため検証はできない) ===")
    X_test = preprocess_pipeline.transform(test_data)
    forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    forest_clf.fit(X_train, y_train)
    y_pred = forest_clf.predict(X_test)
    logger.info(f"テストデータへの予測件数: {len(y_pred)}, 生存予測の割合: {y_pred.mean():.1%}")

    feature_engineering_glance(train_data)

    logger.info(
        f"まとめ: RandomForest={forest_scores.mean():.4f}, SVC={svm_scores.mean():.4f} "
        "(Kaggleのリーダーボード基準では上位数%相当の精度域)"
    )


if __name__ == "__main__":
    main()
