"""取得済みリーダーボードCSVから、投手を先発/リリーフに分類し、
球速に差があるかを調べるCLI。

登板数（p_game）と投球回（p_formatted_ip）から1登板あたりの平均投球回を求め、
3回以上なら先発、それ未満ならリリーフとして分類する。分類した2群について
指定した球速フィールド（デフォルトは4シーム平均球速）の分布を箱ひげ図で
比較し、平均の差が統計的に有意かをWelchのt検定で確認する。

図中のテキストは英語表記にしている（overview.py等と同様、実行環境にCJK
対応フォントが入っておらず、日本語だとタイトル等が文字化けするため）。
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_DIR))
from data_produce import savant_client
from data_produce.savant_client import SavantField
from data_produce.util.file_io import read_csv, save_figure
from data_produce.util.logging_config import setup_logger


app = typer.Typer(help="投手を先発/リリーフに分類し、球速の差を比較するCLI", add_completion=False)

STARTER = "starter"
RELIEVER = "reliever"
STARTER_IP_PER_GAME = 3.0

# dataviz skillの検証済みパレット（ライトモード）からの抜粋。overview.py等と共通。
# カテゴリ色は固定順の1番目（starter）・2番目（reliever）を使用する。
COLOR_PRIMARY = "#2a78d6"
COLOR_SECONDARY = "#eb6834"
COLOR_INK = "#0b0b0b"
COLOR_SECONDARY_INK = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"
COLOR_SURFACE = "#fcfcfb"

ROLE_COLORS = {STARTER: COLOR_PRIMARY, RELIEVER: COLOR_SECONDARY}


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


def _classify_role(data: pd.DataFrame, game_col: str, ip_col: str) -> pd.DataFrame:
    """1登板あたりの平均投球回から、先発/リリーフを分類する。"""

    data[ip_col] = _to_decimal_innings(data[ip_col])
    data["ip_per_game"] = data[ip_col] / data[game_col]
    data["role"] = np.where(data["ip_per_game"] >= STARTER_IP_PER_GAME, STARTER, RELIEVER)
    return data


def _plot_role_comparison(
    data: pd.DataFrame,
    velocity_col: str,
    year: int,
    t_stat: float,
    p_value: float,
) -> plt.Figure:
    groups = [data.loc[data["role"] == role, velocity_col] for role in (STARTER, RELIEVER)]
    labels = [f"{role} (n={len(g)})" for role, g in zip((STARTER, RELIEVER), groups)]

    fig, ax = plt.subplots(figsize=(7, 6))
    box = ax.boxplot(
        groups,
        tick_labels=labels,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": COLOR_INK, "linewidth": 1.5},
        whiskerprops={"color": COLOR_MUTED},
        capprops={"color": COLOR_MUTED},
        flierprops={"markeredgecolor": COLOR_MUTED, "markersize": 4},
    )
    for patch, role in zip(box["boxes"], (STARTER, RELIEVER)):
        patch.set_facecolor(ROLE_COLORS[role])
        patch.set_alpha(0.35)
        patch.set_edgecolor(ROLE_COLORS[role])

    rng = np.random.default_rng(0)
    for i, (role, g) in enumerate(zip((STARTER, RELIEVER), groups), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(g))
        ax.scatter(
            np.full(len(g), i) + jitter,
            g,
            color=ROLE_COLORS[role],
            alpha=0.6,
            edgecolor=COLOR_SURFACE,
            linewidth=0.3,
            s=18,
            zorder=3,
        )

    ax.set_title(
        f"{year} pitcher / {velocity_col} by role (t={t_stat:.2f}, p={p_value:.3f})",
        loc="left",
    )
    ax.set_ylabel(velocity_col)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


@app.command()
def role(
    year: int = typer.Option(2026, "--year", "-y", help="対象の年"),
    velocity_field: SavantField = typer.Option(
        SavantField.FASTBALL_VELOCITY, "--velocity-field", help="比較する球速フィールド"
    ),
) -> None:
    """投手を先発/リリーフに分類し、球速の差を箱ひげ図とWelchのt検定で比較する。

    事前に`data_produce/fetch_leaderboard.py`で`--field p_game --field
    p_formatted_ip --field <velocity_field>`の組み合わせを取得しておく必要がある。
    「q」（規定投球回）フィルタだとリリーフ投手の大半が除外されるため、
    `--min-sample`に十分小さい値（例: `40`）を指定すること。
    """

    logger = setup_logger(Path(__file__).stem)
    _apply_style()

    fields = [SavantField.GAMES, SavantField.INNINGS_PITCHED, velocity_field]
    stat_key = savant_client.stat_key_for(fields)
    csv_name = f"leaderboard/pitcher/{year}/{stat_key}.csv"
    df = read_csv(csv_name)

    game_col = SavantField.GAMES.value
    ip_col = SavantField.INNINGS_PITCHED.value
    velocity_col = velocity_field.value

    data = df[[game_col, ip_col, velocity_col]].dropna().copy()
    data = _classify_role(data, game_col, ip_col)

    starter_v = data.loc[data["role"] == STARTER, velocity_col]
    reliever_v = data.loc[data["role"] == RELIEVER, velocity_col]
    t_stat, p_value = stats.ttest_ind(starter_v, reliever_v, equal_var=False)

    logger.info(f"=== [{stat_key}] {csv_name} ===")
    logger.info(f"先発: n={len(starter_v)} 平均{velocity_col}={starter_v.mean():.2f}")
    logger.info(f"リリーフ: n={len(reliever_v)} 平均{velocity_col}={reliever_v.mean():.2f}")
    logger.info(f"Welchのt検定: t={t_stat:.3f} p={p_value:.4f}")

    fig = _plot_role_comparison(data, velocity_col, year, t_stat, p_value)
    fig_path = save_figure(fig, f"pitcher/{year}/{stat_key}_role_comparison.png", dpi=150)
    plt.close(fig)

    logger.info(f"図を保存しました: {fig_path}")


if __name__ == "__main__":
    app()
