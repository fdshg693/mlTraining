"""取得済みリーダーボードCSVから、単回帰でフィールド間の値を予測するCLI。

`correlation.py`で関係を確認した2つのフィールドについて、片方の実測値
（例: 球速）からもう片方（例: スピン）を線形回帰で予測し、区間つきで示す。

区間は2種類算出する。
- 信頼区間（confidence interval）: 「その説明変数の値における平均応答」がどの
  範囲に収まるか。同じ説明変数の値を持つ選手全体の平均スピンの区間
- 予測区間（prediction interval）: 「その説明変数の値を持つ、まだ見ぬ1件の
  観測値」がどの範囲に収まるか。信頼区間より必ず広くなる

「実際に球速が分かったとき、その投手のスピンがどの範囲に収まるか」を知り
たい場合は、予測区間の方を見る。

図中のテキストは英語表記にしている（overview.py/correlation.pyと同様、実行環境に
CJK対応フォントが入っておらず、日本語だとタイトル等が文字化けするため）。
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
from sklearn.linear_model import LinearRegression

PROJECT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_DIR))
from data_produce import savant_client
from data_produce.savant_client import SavantField
from data_produce.util.file_io import read_csv, save_figure, save_table
from data_produce.util.logging_config import setup_logger


app = typer.Typer(
    help="単回帰で一方のフィールドの実測値からもう一方を予測し、信頼区間・予測区間を算出するCLI",
    add_completion=False,
)

# dataviz skillの検証済みパレット（ライトモード）からの抜粋。overview.py/correlation.pyと共通。
COLOR_PRIMARY = "#2a78d6"
COLOR_INK = "#0b0b0b"
COLOR_SECONDARY_INK = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"
COLOR_SURFACE = "#fcfcfb"
COLOR_TARGET = "#c0392b"


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


def _fit_ols(data: pd.DataFrame, x_col: str, y_col: str) -> dict:
    """単回帰をあてはめ、区間算出に必要な統計量をまとめて返す。"""

    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)
    n = len(x)
    dof = n - 2
    if dof < 1:
        raise typer.BadParameter(f"回帰の自由度が不足しています（n={n}）。データ数を確認してください")

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    residuals = y - model.predict(x.reshape(-1, 1))
    residual_std_error = np.sqrt(np.sum(residuals**2) / dof)
    x_mean = x.mean()
    sxx = np.sum((x - x_mean) ** 2)

    return {
        "slope": slope,
        "intercept": intercept,
        "n": n,
        "dof": dof,
        "residual_std_error": residual_std_error,
        "x_mean": x_mean,
        "sxx": sxx,
        "r2": model.score(x.reshape(-1, 1), y),
    }


def _predict_with_intervals(fit: dict, x_new: np.ndarray, confidence: float) -> pd.DataFrame:
    """指定した説明変数の値それぞれについて、予測値・信頼区間・予測区間を算出する。"""

    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df=fit["dof"])

    y_pred = fit["intercept"] + fit["slope"] * x_new
    # leverage(てこ比) h(x0) = 1/n + (x0 - x_mean)^2 / Sxx
    # 第1項 1/n: 標本平均の不確実性 Var(y_mean) = residual_std_error^2 / n に由来し、nが大きいほど小さくなる
    # 第2項: 予測したいx_newが回帰に使ったxの平均から離れるほど大きくなる（回帰直線はx_meanを軸に誤差が広がるため）
    # この値に residual_std_error を掛けたものが信頼区間・予測区間の幅の元になる（se_mean, se_indivを参照）
    leverage = 1 / fit["n"] + (x_new - fit["x_mean"]) ** 2 / fit["sxx"]
    se_mean = fit["residual_std_error"] * np.sqrt(leverage)
    se_indiv = fit["residual_std_error"] * np.sqrt(1 + leverage)
    ci_half = t_crit * se_mean
    pi_half = t_crit * se_indiv

    return pd.DataFrame(
        {
            "x": x_new,
            "y_pred": y_pred,
            "ci_low": y_pred - ci_half,
            "ci_high": y_pred + ci_half,
            "pi_low": y_pred - pi_half,
            "pi_high": y_pred + pi_half,
        }
    )


def _plot_prediction(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    player_type: str,
    year: int,
    fit: dict,
    targets: pd.DataFrame,
    confidence: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        data[x_col],
        data[y_col],
        color=COLOR_PRIMARY,
        alpha=0.6,
        edgecolor=COLOR_SURFACE,
        linewidth=0.4,
        label="observed",
        zorder=2,
    )

    x_min = min(data[x_col].min(), targets["x"].min())
    x_max = max(data[x_col].max(), targets["x"].max())
    x_pad = (x_max - x_min) * 0.05 or 1.0
    x_line = np.linspace(x_min - x_pad, x_max + x_pad, 200)
    band = _predict_with_intervals(fit, x_line, confidence)

    pct = int(round(confidence * 100))
    ax.plot(x_line, band["y_pred"], color=COLOR_INK, linewidth=1.5, label="regression line", zorder=3)
    ax.fill_between(
        x_line, band["pi_low"], band["pi_high"], color=COLOR_MUTED, alpha=0.18, label=f"{pct}% prediction interval", zorder=1
    )
    ax.fill_between(
        x_line, band["ci_low"], band["ci_high"], color=COLOR_PRIMARY, alpha=0.25, label=f"{pct}% confidence interval", zorder=1
    )

    ax.errorbar(
        targets["x"],
        targets["y_pred"],
        yerr=[targets["y_pred"] - targets["pi_low"], targets["pi_high"] - targets["y_pred"]],
        fmt="o",
        color=COLOR_TARGET,
        ecolor=COLOR_TARGET,
        elinewidth=1.5,
        capsize=4,
        label="target prediction",
        zorder=4,
    )

    ax.set_title(
        f"{year} {player_type} / {x_col} -> {y_col} prediction (r2={fit['r2']:.2f}, n={fit['n']})",
        loc="left",
        fontsize=10,
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_axisbelow(True)
    ax.grid(linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


@app.command()
def predict(
    year: int = typer.Option(2026, "--year", "-y", help="対象の年"),
    player_type: str = typer.Option(
        "pitcher", "--type", "-t", help="対象（pitcher または batter）"
    ),
    x_field: SavantField = typer.Option(
        SavantField.FASTBALL_VELOCITY, "--x-field", help="説明変数（実測値が分かっている側）のフィールド"
    ),
    y_field: SavantField = typer.Option(
        SavantField.FASTBALL_SPIN, "--y-field", help="目的変数（予測したい側）のフィールド"
    ),
    speed: list[float] = typer.Option(
        ..., "--speed", "-s", help="予測したいx-fieldの実測値。複数指定可（-s 96.5 -s 98.0）"
    ),
    confidence: float = typer.Option(
        0.95, "--confidence", "-c", help="信頼区間・予測区間の信頼度（0〜1、デフォルト0.95）"
    ),
) -> None:
    """指定した年・対象について、x-fieldの実測値からy-fieldを単回帰で予測し、
    信頼区間（平均応答）と予測区間（個別の新規観測値）を算出して保存する。

    事前に `data_produce/fetch_leaderboard.py` で `--field x-field --field y-field`
    の組み合わせを取得しておく必要がある。
    """

    if player_type not in ("pitcher", "batter"):
        raise typer.BadParameter("typeは'pitcher'か'batter'を指定してください")
    if x_field == y_field:
        raise typer.BadParameter("x-fieldとy-fieldには異なるフィールドを指定してください")
    if not 0 < confidence < 1:
        raise typer.BadParameter("confidenceは0〜1の範囲で指定してください")

    logger = setup_logger(Path(__file__).stem)
    _apply_style()

    stat_key = savant_client.stat_key_for([x_field, y_field])
    csv_name = f"leaderboard/{player_type}/{year}/{stat_key}.csv"
    df = read_csv(csv_name)

    x_col, y_col = x_field.value, y_field.value
    data = df[[x_col, y_col]].dropna()

    fit = _fit_ols(data, x_col, y_col)
    targets = _predict_with_intervals(fit, np.array(speed, dtype=float), confidence)

    pct = int(round(confidence * 100))
    logger.info(f"=== [{stat_key}] {csv_name} ===")
    logger.info(
        f"回帰式: {y_col} = {fit['intercept']:.3f} + {fit['slope']:.3f} * {x_col} "
        f"(r2={fit['r2']:.3f}, n={fit['n']})"
    )
    for _, row in targets.iterrows():
        logger.info(
            f"{x_col}={row['x']:.2f} → {y_col}予測値={row['y_pred']:.2f} / "
            f"{pct}%信頼区間(平均) [{row['ci_low']:.2f}, {row['ci_high']:.2f}] / "
            f"{pct}%予測区間(個別) [{row['pi_low']:.2f}, {row['pi_high']:.2f}]"
        )

    fig = _plot_prediction(data, x_col, y_col, player_type, year, fit, targets, confidence)
    fig_path = save_figure(fig, f"{player_type}/{year}/{stat_key}_predict.png", dpi=150)
    plt.close(fig)

    table = targets.rename(columns={"x": x_col, "y_pred": f"{y_col}_pred"})
    table_path = save_table(table, f"{player_type}/{year}/{stat_key}_predict.csv")

    logger.info(f"図を保存しました: {fig_path}")
    logger.info(f"表を保存しました: {table_path}")


if __name__ == "__main__":
    app()
