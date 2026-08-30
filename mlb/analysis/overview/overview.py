"""取得済みリーダーボードCSVの概要（統計サマリ＋分布・ランキング図）を表示するCLI。

図中のテキストは英語表記にしている。実行環境にCJK対応フォントが
入っておらず、日本語だとタイトル等が文字化け（豆腐）するため。
ログ出力はプロジェクトの規約通り日本語のまま。
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import typer

PROJECT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_DIR))
from data_produce import savant_client
from data_produce.util.file_io import read_csv, save_figure
from data_produce.util.logging_config import setup_logger


app = typer.Typer(help="取得済みリーダーボードCSVの概要を表示するCLI", add_completion=False)

NAME_COL = "last_name, first_name"

# dataviz skillの検証済みパレット（ライトモード）からの抜粋。
COLOR_PRIMARY = "#2a78d6"
COLOR_INK = "#0b0b0b"
COLOR_SECONDARY_INK = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"
COLOR_SURFACE = "#fcfcfb"


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


def _print_df_summary(df: pd.DataFrame, logger) -> None:
    logger.info(f"行数 x 列数: {df.shape[0]:,} x {df.shape[1]}")
    logger.info(f"列名: {list(df.columns)}")

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        logger.info("欠損値: なし")
    else:
        for col, count in missing.items():
            logger.info(f"欠損値: {col} = {count}件 ({count / len(df):.1%})")

    logger.info(f"重複行: {df.duplicated().sum()}件")


def _print_value_summary(df: pd.DataFrame, value_col: str, logger) -> None:
    stats = df[value_col].describe()
    logger.info(
        f"[{value_col}] 件数={stats['count']:.0f} 平均={stats['mean']:.2f} "
        f"標準偏差={stats['std']:.2f} 最小={stats['min']:.2f} "
        f"中央値={stats['50%']:.2f} 最大={stats['max']:.2f}"
    )


def _plot_distribution(
    df: pd.DataFrame, value_col: str, player_type: str, year: int
) -> plt.Figure:
    data = df[value_col].dropna()
    mean = data.mean()
    median = data.median()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=20, color=COLOR_PRIMARY, edgecolor=COLOR_SURFACE, linewidth=0.5)
    ax.axvline(mean, color=COLOR_INK, linestyle="--", linewidth=1.5, label=f"mean {mean:.1f}")
    ax.axvline(
        median, color=COLOR_SECONDARY_INK, linestyle=":", linewidth=1.5, label=f"median {median:.1f}"
    )
    ax.set_title(f"{year} {player_type} / {value_col} distribution (n={len(data)})", loc="left")
    ax.set_xlabel(value_col)
    ax.set_ylabel("players")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _plot_ranking(
    df: pd.DataFrame,
    value_col: str,
    player_type: str,
    year: int,
    top_n: int,
) -> plt.Figure:
    ranked = df[[NAME_COL, value_col]].dropna().sort_values(value_col, ascending=False).head(top_n)
    ranked = ranked.iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(ranked[NAME_COL], ranked[value_col], color=COLOR_PRIMARY, height=0.6)
    for y, value in enumerate(ranked[value_col]):
        ax.text(value, y, f" {value:.1f}", va="center", color=COLOR_SECONDARY_INK, fontsize=9)
    ax.set_title(f"{year} {player_type} / {value_col} top {top_n}", loc="left")
    ax.set_xlabel(value_col)
    ax.set_axisbelow(True)
    ax.grid(axis="x", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


@app.command()
def overview(
    year: int = typer.Option(2026, "--year", "-y", help="対象の年"),
    player_type: str = typer.Option(
        "pitcher", "--type", "-t", help="対象（pitcher または batter）"
    ),
    field: list[savant_client.SavantField] = typer.Option(
        [savant_client.SavantField.FASTBALL_VELOCITY],
        "--field",
        "-f",
        help="概要を表示するフィールド（複数指定可。組み合わせに応じたCSVを読み込む）",
    ),
    top_n: int = typer.Option(15, "--top-n", help="ランキング図に表示する人数"),
) -> None:
    """指定した年・対象・フィールドの組み合わせのリーダーボードCSVについて、統計サマリと分布・ランキング図を出力する。"""

    if player_type not in ("pitcher", "batter"):
        raise typer.BadParameter("typeは'pitcher'か'batter'を指定してください")

    logger = setup_logger(Path(__file__).stem)
    _apply_style()

    stat_key = savant_client.stat_key_for(field)
    csv_name = f"leaderboard/{player_type}/{year}/{stat_key}.csv"
    df = read_csv(csv_name)

    logger.info(f"=== [{stat_key}] {csv_name} ===")
    _print_df_summary(df, logger)

    out_dir = f"{player_type}/{year}"

    for f in field:
        value_col = f.value
        _print_value_summary(df, value_col, logger)

        dist_fig = _plot_distribution(df, value_col, player_type, year)
        dist_path = save_figure(dist_fig, f"{out_dir}/{value_col}_distribution.png", dpi=150)
        plt.close(dist_fig)

        rank_fig = _plot_ranking(df, value_col, player_type, year, top_n=top_n)
        rank_path = save_figure(rank_fig, f"{out_dir}/{value_col}_ranking.png", dpi=150)
        plt.close(rank_fig)

        logger.info(f"[{value_col}] 図を保存しました: {dist_path} / {rank_path}")


if __name__ == "__main__":
    app()
