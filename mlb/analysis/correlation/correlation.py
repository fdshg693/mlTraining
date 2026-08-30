"""取得済みリーダーボードCSVから、2つのフィールド間の相関関係を調べるCLI。

図中のテキストは英語表記にしている（overview.pyと同様、実行環境にCJK対応
フォントが入っておらず、日本語だとタイトル等が文字化けするため）。
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

PROJECT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_DIR))
from data_produce import savant_client
from data_produce.savant_client import SavantField
from data_produce.util.file_io import read_csv, save_figure
from data_produce.util.logging_config import setup_logger


app = typer.Typer(help="2つのフィールド間の相関関係を調べるCLI", add_completion=False)

# dataviz skillの検証済みパレット（ライトモード）からの抜粋。overview.pyと共通。
COLOR_PRIMARY = "#2a78d6"
COLOR_INK = "#0b0b0b"
COLOR_SECONDARY_INK = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"
COLOR_SURFACE = "#fcfcfb"


def _to_decimal_innings(series: pd.Series) -> pd.Series:
    """savantの投球回表記（例: 125.2 = 125回2/3）を実際の小数のイニング数に変換する。

    小数点以下は10進の端数ではなく、アウト数（0/1/2）を表す独自表記のため。
    """

    whole = np.floor(series)
    outs = np.round((series - whole) * 10)
    return whole + outs / 3


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


def _plot_scatter(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    player_type: str,
    year: int,
    corr: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        data[x_col], data[y_col], color=COLOR_PRIMARY, alpha=0.7, edgecolor=COLOR_SURFACE, linewidth=0.4
    )

    slope, intercept = np.polyfit(data[x_col], data[y_col], 1)
    x_line = np.linspace(data[x_col].min(), data[x_col].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color=COLOR_INK, linestyle="--", linewidth=1.5)

    ax.set_title(f"{year} {player_type} / {x_col} vs {y_col} (r={corr:.2f}, n={len(data)})", loc="left")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_axisbelow(True)
    ax.grid(linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


@app.command()
def correlation(
    year: int = typer.Option(2026, "--year", "-y", help="対象の年"),
    player_type: str = typer.Option(
        "pitcher", "--type", "-t", help="対象（pitcher または batter）"
    ),
    x_field: SavantField = typer.Option(
        SavantField.FASTBALL_VELOCITY, "--x-field", help="横軸のフィールド"
    ),
    y_field: SavantField = typer.Option(
        SavantField.FASTBALL_SPIN, "--y-field", help="縦軸のフィールド"
    ),
) -> None:
    """指定した年・対象について、2つのフィールド間の相関係数を算出し散布図を保存する。

    事前に `data_produce/fetch_leaderboard.py` で `--field x-field --field y-field`
    の組み合わせを取得しておく必要がある。
    """

    if player_type not in ("pitcher", "batter"):
        raise typer.BadParameter("typeは'pitcher'か'batter'を指定してください")
    if x_field == y_field:
        raise typer.BadParameter("x-fieldとy-fieldには異なるフィールドを指定してください")

    logger = setup_logger(Path(__file__).stem)
    _apply_style()

    stat_key = savant_client.stat_key_for([x_field, y_field])
    csv_name = f"leaderboard/{player_type}/{year}/{stat_key}.csv"
    df = read_csv(csv_name)

    x_col, y_col = x_field.value, y_field.value
    data = df[[x_col, y_col]].dropna().copy()
    if x_field == SavantField.INNINGS_PITCHED:
        data[x_col] = _to_decimal_innings(data[x_col])
    if y_field == SavantField.INNINGS_PITCHED:
        data[y_col] = _to_decimal_innings(data[y_col])
    corr = data[x_col].corr(data[y_col])

    logger.info(f"=== [{stat_key}] {csv_name} ===")
    logger.info(f"{x_col} と {y_col} の相関係数: r={corr:.3f} (n={len(data)})")

    fig = _plot_scatter(data, x_col, y_col, player_type, year, corr)
    fig_path = save_figure(fig, f"{player_type}/{year}/{stat_key}_scatter.png", dpi=150)
    plt.close(fig)

    logger.info(f"図を保存しました: {fig_path}")


if __name__ == "__main__":
    app()
