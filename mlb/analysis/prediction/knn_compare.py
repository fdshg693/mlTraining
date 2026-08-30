"""取得済みリーダーボードCSVから、線形回帰とKNNの予測を比較するCLI。

`predict.py`は線形回帰（モデルベース学習）のみを扱っている。このスクリプトは
同じ`x_field`/`y_field`ペアに`KNeighborsRegressor`（インスタンスベース学習）
も当てはめ、線形回帰の予測直線とKNNの予測曲線を重ねて可視化する。

`k`（近傍数）を複数指定すると、handson-ml3の
`experiment/knn_k_sweep.py`と同様に、kが小さいほど訓練データに沿った
ガタガタの曲線（過学習）、kが大きいほど全体平均に寄ったなめらかな曲線
（未学習）になる様子を1枚の図で観察できる。R^2はいずれも全データで
学習・評価した値（`predict.py`と同様、train/test分割はしていない。
汎化性能を厳密に比較したい場合は`generalization.py`を参照）。特に
k=1はほぼ全ての点を通るため、訓練データ上のR^2は1.0に近くなりやすいが、
これは汎化性能の高さを意味しない。

図中のテキストは英語表記にしている（predict.py等と同様、実行環境に
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
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

PROJECT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_DIR))
from data_produce import savant_client
from data_produce.savant_client import SavantField
from data_produce.util.file_io import read_csv, save_figure, save_table
from data_produce.util.logging_config import setup_logger


app = typer.Typer(
    help="同じフィールドペアに線形回帰とKNNを当てはめ、予測直線/曲線を重ねて比較するCLI",
    add_completion=False,
)

# dataviz skillの検証済みパレット（ライトモード）からの抜粋。predict.py/generalization.pyと共通。
COLOR_INK = "#0b0b0b"
COLOR_SECONDARY_INK = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"
COLOR_SURFACE = "#fcfcfb"
# カテゴリカル配色スロット1〜8（固定順）。KNNのk違いをこの順で割り当てる。
CATEGORICAL_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]


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


def _plot_comparison(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    player_type: str,
    year: int,
    linear_model: LinearRegression,
    r2_linear: float,
    knn_results: list[tuple[int, KNeighborsRegressor, float]],
    targets: list[float],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        data[x_col],
        data[y_col],
        color=COLOR_MUTED,
        alpha=0.5,
        edgecolor=COLOR_SURFACE,
        linewidth=0.4,
        s=20,
        label="observed",
        zorder=2,
    )

    x_min = min(data[x_col].min(), *targets) if targets else data[x_col].min()
    x_max = max(data[x_col].max(), *targets) if targets else data[x_col].max()
    x_pad = (x_max - x_min) * 0.05 or 1.0
    x_line = np.linspace(x_min - x_pad, x_max + x_pad, 300).reshape(-1, 1)

    ax.plot(
        x_line[:, 0],
        linear_model.predict(x_line),
        color=COLOR_INK,
        linewidth=1.8,
        linestyle="--",
        label=f"linear regression (R2={r2_linear:.2f})",
        zorder=4,
    )
    for (k, model, r2_knn), color in zip(knn_results, CATEGORICAL_COLORS):
        ax.plot(
            x_line[:, 0],
            model.predict(x_line),
            color=color,
            linewidth=1.6,
            label=f"KNN k={k} (R2={r2_knn:.2f})",
            zorder=3,
        )

    for target in targets:
        ax.axvline(target, color=COLOR_BASELINE, linestyle=":", linewidth=1.0, zorder=1)

    ax.set_title(
        f"{year} {player_type} / {x_col} -> {y_col}: linear vs KNN",
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
def knn_compare(
    year: int = typer.Option(2026, "--year", "-y", help="対象の年"),
    player_type: str = typer.Option(
        "pitcher", "--type", "-t", help="対象（pitcher または batter）"
    ),
    x_field: SavantField = typer.Option(
        SavantField.FASTBALL_VELOCITY, "--x-field", help="説明変数のフィールド"
    ),
    y_field: SavantField = typer.Option(
        SavantField.FASTBALL_SPIN, "--y-field", help="目的変数のフィールド"
    ),
    k: list[int] = typer.Option(
        [1, 5, 15, 40], "--k", help="比較するKNNの近傍数(n_neighbors)。複数指定可（--k 1 --k 5 ...）"
    ),
    target: list[float] = typer.Option(
        [], "--target", "-x", help="予測値を比較したいx-fieldの値。複数指定可。省略時は予測値のログ・マーカー表示をしない"
    ),
) -> None:
    """指定した年・対象について、x-fieldからy-fieldへの線形回帰とKNN（複数k）を
    同じデータに当てはめ、予測直線/曲線を重ねた図と予測値の表を保存する。

    事前に `data_produce/fetch_leaderboard.py` で `--field x-field --field y-field`
    の組み合わせを取得しておく必要がある。
    """

    if player_type not in ("pitcher", "batter"):
        raise typer.BadParameter("typeは'pitcher'か'batter'を指定してください")
    if x_field == y_field:
        raise typer.BadParameter("x-fieldとy-fieldには異なるフィールドを指定してください")
    if not k:
        raise typer.BadParameter("kは1つ以上指定してください")
    if len(set(k)) != len(k):
        raise typer.BadParameter("kに重複した値が含まれています")
    if any(kk < 1 for kk in k):
        raise typer.BadParameter("kは1以上の整数を指定してください")
    if len(k) > len(CATEGORICAL_COLORS):
        raise typer.BadParameter(f"kは{len(CATEGORICAL_COLORS)}個以内で指定してください（配色の都合）")

    logger = setup_logger(Path(__file__).stem)
    _apply_style()

    stat_key = savant_client.stat_key_for([x_field, y_field])
    csv_name = f"leaderboard/{player_type}/{year}/{stat_key}.csv"
    df = read_csv(csv_name)

    x_col, y_col = x_field.value, y_field.value
    data = df[[x_col, y_col]].dropna()
    X = data[[x_col]].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)

    if len(X) <= max(k):
        raise typer.BadParameter(f"データ数（n={len(X)}）が最大のk（{max(k)}）以下です。kを小さくするかデータを増やしてください")

    linear_model = LinearRegression().fit(X, y)
    r2_linear = linear_model.score(X, y)

    knn_results = []
    for kk in k:
        knn_model = KNeighborsRegressor(n_neighbors=kk).fit(X, y)
        knn_results.append((kk, knn_model, knn_model.score(X, y)))

    logger.info(f"=== [{stat_key}] {csv_name} (n={len(X)}) ===")
    logger.info(f"線形回帰（モデルベース学習）: R^2={r2_linear:.3f}")
    for kk, _, r2_knn in knn_results:
        note = "（訓練データを丸暗記するためR^2は1.0に近くなりやすい＝過学習の兆候）" if kk == 1 else ""
        logger.info(f"KNN k={kk}（インスタンスベース学習）: R^2={r2_knn:.3f}{note}")

    for xt in target:
        preds = [f"linear: {linear_model.predict([[xt]])[0]:.2f}"]
        preds += [f"knn(k={kk}): {model.predict([[xt]])[0]:.2f}" for kk, model, _ in knn_results]
        logger.info(f"{x_col}={xt:.2f} → {y_col}予測値 " + " / ".join(preds))

    fig = _plot_comparison(data, x_col, y_col, player_type, year, linear_model, r2_linear, knn_results, target)
    fig_path = save_figure(fig, f"{player_type}/{year}/{stat_key}_knn_compare.png", dpi=150)
    plt.close(fig)

    x_line = np.linspace(data[x_col].min(), data[x_col].max(), 300).reshape(-1, 1)
    table = pd.DataFrame({x_col: x_line[:, 0], "linear_pred": linear_model.predict(x_line)})
    for kk, model, _ in knn_results:
        table[f"knn_k{kk}_pred"] = model.predict(x_line)
    table_path = save_table(table, f"{player_type}/{year}/{stat_key}_knn_compare.csv")

    logger.info(f"図を保存しました: {fig_path}")
    logger.info(f"表を保存しました: {table_path}")


if __name__ == "__main__":
    app()
