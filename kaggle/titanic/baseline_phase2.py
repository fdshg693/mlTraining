"""フェーズ2: ベースライン構築。

handson-ml3の演習(exercise/04_titanic.py)で使った前処理パイプライン(数値: 欠損補完+
標準化、カテゴリ: 欠損補完+OneHotEncoding)を土台に、LogisticRegression/
RandomForestClassifier/SVCを同条件の10-fold交差検証で比較する。最良モデルを
train.csv全体で学習し直し、test.csvへの予測をgender_submission.csvと同じ列構成
(PassengerId, Survived)で出力する(提出用ファイル生成の型を確立する)。

図中のテキストは英語表記にしている(eda_phase1.pyと同じ理由。実行環境にCJK対応
フォントが入っておらず、日本語だと文字化けするため)。ログ出力は日本語のまま。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

PROJECT_DIR = Path(__file__).resolve().parents[2]
KAGGLE_DATA_DIR = PROJECT_DIR / "kaggle" / "data" / "titanic"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
LOG_DIR = Path(__file__).resolve().parent / "logs"

# dataviz skillの検証済みパレット(ライトモード)からの抜粋。eda_phase1.pyと共通。
COLOR_PRIMARY = "#2a78d6"  # categorical slot 1: blue
COLOR_INK = "#0b0b0b"
COLOR_SECONDARY_INK = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"
COLOR_SURFACE = "#fcfcfb"

NUM_ATTRIBS = ["Age", "SibSp", "Parch", "Fare"]
CAT_ATTRIBS = ["Pclass", "Sex", "Embarked"]
CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


def _setup_logger() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        LOG_DIR / "baseline_phase2.log",
        mode="w",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )
    logger.add(lambda message: print(message, end=""), format="{message}")


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLOR_SURFACE,
            "axes.facecolor": COLOR_SURFACE,
            "axes.edgecolor": COLOR_BASELINE,
            "axes.labelcolor": COLOR_SECONDARY_INK,
            "text.color": COLOR_INK,
            "xtick.color": COLOR_MUTED,
            "ytick.color": COLOR_MUTED,
            "grid.color": COLOR_GRID,
            "font.size": 11,
        }
    )


def _clean_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _save_figure(fig: plt.Figure, file_name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / file_name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"図を保存しました: {path}")
    return path


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(KAGGLE_DATA_DIR / "train.csv")
    test = pd.read_csv(KAGGLE_DATA_DIR / "test.csv")
    logger.info(f"train.shape={train.shape}, test.shape={test.shape}")
    return train, test


def build_preprocess_pipeline() -> ColumnTransformer:
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipeline, NUM_ATTRIBS),
            ("cat", cat_pipeline, CAT_ATTRIBS),
        ]
    )


def build_candidate_models() -> dict[str, Pipeline]:
    return {
        "LogisticRegression": Pipeline(
            [
                ("preprocess", build_preprocess_pipeline()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("preprocess", build_preprocess_pipeline()),
                ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ]
        ),
        "SVC": Pipeline(
            [
                ("preprocess", build_preprocess_pipeline()),
                ("clf", SVC(gamma="auto")),
            ]
        ),
    }


def compare_models(
    models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    logger.info(f"=== モデル比較({CV.get_n_splits()}-fold交差検証, Accuracy) ===")
    rows = []
    for name, pipeline in models.items():
        scores = cross_val_score(pipeline, X, y, cv=CV, scoring="accuracy")
        rows.append({"model": name, "mean": scores.mean(), "std": scores.std()})
        logger.info(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")

    result = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    logger.info(
        f"最良モデル: {result.iloc[0]['model']}(mean={result.iloc[0]['mean']:.4f})。"
        "以降の実験(フェーズ3以降)はこのスコアを比較対象とする"
    )
    return result


def plot_model_comparison(result: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        result["model"],
        result["mean"] * 100,
        yerr=result["std"] * 100,
        color=COLOR_PRIMARY,
        edgecolor=COLOR_SURFACE,
        capsize=4,
    )
    ax.set_ylabel("CV accuracy (%)")
    ax.set_title(f"Model comparison ({CV.get_n_splits()}-fold CV)", loc="left")
    ax.set_ylim(0, 100)
    _clean_axes(ax)
    _save_figure(fig, "model_comparison.png")


def make_submission(
    best_model_name: str,
    models: dict[str, Pipeline],
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    logger.info(f"=== 提出用ファイル生成({best_model_name}をtrain全体で再学習) ===")
    pipeline = models[best_model_name]
    pipeline.fit(train[NUM_ATTRIBS + CAT_ATTRIBS], train["Survived"])
    y_pred = pipeline.predict(test[NUM_ATTRIBS + CAT_ATTRIBS])

    submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": y_pred})
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "submission_phase2.csv"
    submission.to_csv(path, index=False)
    logger.info(
        f"提出用ファイルを保存しました: {path}(件数={len(submission)}, "
        f"生存予測の割合={y_pred.mean():.1%})"
    )


def main() -> None:
    _setup_logger()
    _apply_style()

    train, test = load_data()
    X_train = train[NUM_ATTRIBS + CAT_ATTRIBS]
    y_train = train["Survived"]

    models = build_candidate_models()
    result = compare_models(models, X_train, y_train)
    plot_model_comparison(result)

    best_model_name = result.iloc[0]["model"]
    make_submission(best_model_name, models, train, test)

    logger.info(
        "=== まとめ: フェーズ2完了。数値(Age/SibSp/Parch/Fare)+カテゴリ"
        "(Pclass/Sex/Embarked)のみのベースラインで3モデルを比較し、"
        f"最良モデル({best_model_name})で提出用ファイルの生成まで通した。"
        "フェーズ3(特徴量エンジニアリング)でこのスコアを上回れるか検証する ==="
    )


if __name__ == "__main__":
    main()
