"""フェーズ1: EDA・データ理解。

train.csv/test.csvの欠損状況、各特徴量とSurvivedの関係、Name/Ticket/Cabinから
抽出できそうな情報(敬称・デッキ・同一チケットグループ)を確認する。

図中のテキストは英語表記にしている(実行環境にCJK対応フォントが入っておらず、
日本語だとタイトル等が文字化けするため)。ログ出力はプロジェクトの規約通り
日本語のまま。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

PROJECT_DIR = Path(__file__).resolve().parents[2]
KAGGLE_DATA_DIR = PROJECT_DIR / "kaggle" / "data" / "titanic"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
LOG_DIR = Path(__file__).resolve().parent / "logs"

# dataviz skillの検証済みパレット(ライトモード)からの抜粋。
COLOR_NOT_SURVIVED = "#2a78d6"  # categorical slot 1: blue
COLOR_SURVIVED = "#eb6834"  # categorical slot 2: orange
COLOR_INK = "#0b0b0b"
COLOR_SECONDARY_INK = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"
COLOR_SURFACE = "#fcfcfb"

CATEGORICAL_FEATURES = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]


def _setup_logger() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        LOG_DIR / "eda_phase1.log",
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


def report_missing_values(train: pd.DataFrame, test: pd.DataFrame) -> None:
    logger.info("=== 欠損値の状況 ===")
    shared_cols = [c for c in test.columns if c != "PassengerId"]
    train_missing = train[shared_cols].isna().mean()
    test_missing = test[shared_cols].isna().mean()
    cols_with_missing = [
        c for c in shared_cols if train_missing[c] > 0 or test_missing[c] > 0
    ]

    for col in cols_with_missing:
        logger.info(
            f"{col}: train={train_missing[col]:.1%}({train[col].isna().sum()}件), "
            f"test={test_missing[col]:.1%}({test[col].isna().sum()}件)"
        )
    logger.info(
        "Cabinは欠損が7割超のため、そのまま使うのではなく先頭文字(デッキ)の有無だけ"
        "使う・別特徴量に変換するなどの工夫が必要になりそう"
    )
    logger.info(
        "Age(train 20%/test 21%)は欠損率が近く、単純中央値よりPclass/Titleごとの"
        "中央値補完の方が精度に寄与する可能性がある(フェーズ3で検証)"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(cols_with_missing))
    width = 0.35
    ax.bar(
        [i - width / 2 for i in x],
        [train_missing[c] * 100 for c in cols_with_missing],
        width=width,
        label="train",
        color=COLOR_NOT_SURVIVED,
        edgecolor=COLOR_SURFACE,
    )
    ax.bar(
        [i + width / 2 for i in x],
        [test_missing[c] * 100 for c in cols_with_missing],
        width=width,
        label="test",
        color=COLOR_SURVIVED,
        edgecolor=COLOR_SURFACE,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(cols_with_missing)
    ax.set_ylabel("missing rate (%)")
    ax.set_title("Missing values: train vs test", loc="left")
    ax.legend(frameon=False)
    _clean_axes(ax)
    _save_figure(fig, "missing_values.png")


def plot_categorical_survival(train: pd.DataFrame) -> None:
    logger.info("=== カテゴリ特徴量とSurvivedの関係 ===")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    for ax, feature in zip(axes, CATEGORICAL_FEATURES):
        summary = (
            train.groupby(feature, dropna=False)["Survived"]
            .agg(["mean", "count"])
            .sort_index()
        )
        labels = [str(idx) for idx in summary.index]
        ax.bar(labels, summary["mean"] * 100, color=COLOR_NOT_SURVIVED, edgecolor=COLOR_SURFACE)
        ax.axhline(
            train["Survived"].mean() * 100,
            color=COLOR_SECONDARY_INK,
            linestyle="--",
            linewidth=1,
            label="overall mean",
        )
        ax.set_title(feature, loc="left")
        ax.set_ylabel("survival rate (%)")
        ax.set_ylim(0, 100)
        _clean_axes(ax)

        rate_str = ", ".join(f"{i}={r:.1%}(n={n})" for i, r, n in zip(summary.index, summary["mean"], summary["count"]))
        logger.info(f"{feature}: {rate_str}")

    axes[0].legend(frameon=False, fontsize=9)
    axes[-1].axis("off")
    fig.suptitle("Survival rate by category", x=0.02, ha="left")
    _save_figure(fig, "categorical_survival.png")


def plot_numeric_distributions(train: pd.DataFrame) -> None:
    logger.info("=== 数値特徴量(Age, Fare)とSurvivedの関係 ===")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, feature in zip(axes, ["Age", "Fare"]):
        data = train.dropna(subset=[feature])
        not_survived = data.loc[data["Survived"] == 0, feature]
        survived = data.loc[data["Survived"] == 1, feature]
        bins = 30
        ax.hist(
            not_survived, bins=bins, alpha=0.6, label="not survived",
            color=COLOR_NOT_SURVIVED, density=True,
        )
        ax.hist(
            survived, bins=bins, alpha=0.6, label="survived",
            color=COLOR_SURVIVED, density=True,
        )
        ax.set_title(feature, loc="left")
        ax.set_xlabel(feature)
        ax.set_ylabel("density")
        ax.legend(frameon=False)
        _clean_axes(ax)

        logger.info(
            f"{feature}: not_survived mean={not_survived.mean():.1f}, "
            f"survived mean={survived.mean():.1f}"
        )

    _save_figure(fig, "numeric_distributions.png")


def extract_title(df: pd.DataFrame) -> pd.Series:
    # Nameは"姓, 敬称. 名"の形式("Braund, Mr. Owen Harris")なので、
    # 最初のカンマの後から最初のピリオドの前までを敬称として取り出す。
    title = df["Name"].str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
    common_titles = {"Mr", "Mrs", "Miss", "Master"}
    return title.where(title.isin(common_titles), "Rare")


def extract_deck(df: pd.DataFrame) -> pd.Series:
    return df["Cabin"].str[0].fillna("Unknown")


def add_eda_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Title"] = extract_title(df)
    df["Deck"] = extract_deck(df)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("count")
    return df


def plot_name_ticket_cabin_hints(train: pd.DataFrame) -> None:
    logger.info("=== Name/Ticket/Cabinからの手がかり ===")

    title_counts = train["Title"].value_counts()
    logger.info(f"Title件数: {title_counts.to_dict()}")
    title_survival = train.groupby("Title")["Survived"].mean().sort_values(ascending=False)
    logger.info(f"Title別生存率: {title_survival.to_dict()}")

    deck_survival = (
        train.groupby("Deck", dropna=False)["Survived"].agg(["mean", "count"]).sort_index()
    )
    logger.info(
        "Deck別生存率(Unknown=Cabin欠損): "
        + ", ".join(f"{d}={r:.1%}(n={n})" for d, r, n in zip(deck_survival.index, deck_survival["mean"], deck_survival["count"]))
    )

    ticket_group_survival = train.groupby("TicketGroupSize")["Survived"].agg(["mean", "count"])
    logger.info(
        "同一チケット人数別生存率: "
        + ", ".join(f"{g}人={r:.1%}(n={n})" for g, r, n in zip(ticket_group_survival.index, ticket_group_survival["mean"], ticket_group_survival["count"]))
    )
    logger.info(
        "Titleは性別・敬称の細かい違い(既婚/未婚、子供等)を捉えられ、Deckは欠損が多いが"
        "既知の値では上位クラスとの相関がありそう、TicketGroupSizeはFamilySizeと近い"
        "が非血縁の同行者も拾える点で異なる特徴量になりうる(いずれもフェーズ3で検証)"
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    order = title_survival.index
    ax.bar(order, title_survival.loc[order] * 100, color=COLOR_NOT_SURVIVED, edgecolor=COLOR_SURFACE)
    ax.set_title("Survival rate by Title", loc="left")
    ax.set_ylabel("survival rate (%)")
    _clean_axes(ax)

    ax = axes[1]
    ax.bar(
        [str(i) for i in deck_survival.index],
        deck_survival["mean"] * 100,
        color=COLOR_NOT_SURVIVED,
        edgecolor=COLOR_SURFACE,
    )
    ax.set_title("Survival rate by Deck", loc="left")
    ax.set_ylabel("survival rate (%)")
    _clean_axes(ax)

    ax = axes[2]
    ax.bar(
        [str(i) for i in ticket_group_survival.index],
        ticket_group_survival["mean"] * 100,
        color=COLOR_NOT_SURVIVED,
        edgecolor=COLOR_SURFACE,
    )
    ax.set_title("Survival rate by TicketGroupSize", loc="left")
    ax.set_ylabel("survival rate (%)")
    ax.set_xlabel("same-ticket passenger count")
    _clean_axes(ax)

    _save_figure(fig, "name_ticket_cabin_hints.png")


def save_features_csv(train: pd.DataFrame, test: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "train_eda_features.csv"
    test_path = DATA_DIR / "test_eda_features.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    logger.info(f"抽出済み特徴量つきデータを保存しました: {train_path}, {test_path}")


def main() -> None:
    _setup_logger()
    _apply_style()

    train, test = load_data()

    report_missing_values(train, test)
    plot_categorical_survival(train)
    plot_numeric_distributions(train)

    train = add_eda_features(train)
    test = add_eda_features(test)
    plot_name_ticket_cabin_hints(train)
    save_features_csv(train, test)

    logger.info(
        "=== まとめ: フェーズ1完了。Sex/Pclass/Title/Fareで生存率の差が大きく、"
        "Cabinは欠損対応、TicketGroupSize/FamilySizeは同行者情報として有望。"
        "フェーズ2のベースライン構築に進む ==="
    )


if __name__ == "__main__":
    main()
