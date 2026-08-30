"""取得済みリーダーボードCSVから、多項式回帰の次数スイープを比較するCLI。

`knn_compare.py`はKNNの`k`を変えて過学習・未学習を観察したが、このスクリプトは
`PolynomialFeatures`の`degree`を複数指定し、次数を上げるほど訓練誤差(RMSE)は
下がり続ける一方、テスト誤差はある次数を境にU字に転じる（未学習→ちょうど良い
→過学習）様子を観察する。`knn_compare.py`の`k`を`degree`に置き換えた対になる
スクリプトとして`prediction/`に追加した。

`generalization.py`と同様に`train_test_split`で訓練/テストに分け、各次数の
モデルを訓練データのみで学習する。予測曲線は訓練データにしか触れていないため、
次数が大きいほど訓練点をなぞる一方でテスト点から外れていく様子が図から見える。
`PolynomialFeatures`は単独だと次数が上がるほど特徴量のスケールが桁違いに
広がり数値的に不安定になるため、`degree_sweep.py`（handson-ml3/01）と同様に
`StandardScaler`を挟んだ`Pipeline`にしている。

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
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

PROJECT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_DIR))
from data_produce import savant_client
from data_produce.savant_client import SavantField
from data_produce.util.file_io import read_csv, save_figure, save_table
from data_produce.util.logging_config import setup_logger


app = typer.Typer(
    help="多項式回帰の次数を複数指定し、訓練/テストRMSEと予測曲線から過学習・未学習を比較するCLI",
    add_completion=False,
)

# dataviz skillの検証済みパレット（ライトモード）からの抜粋。knn_compare.py/generalization.pyと共通。
COLOR_INK = "#0b0b0b"
COLOR_SECONDARY_INK = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"
COLOR_SURFACE = "#fcfcfb"
COLOR_TEST = "#eb6834"
# カテゴリカル配色スロット1〜8（固定順）。degree違いをこの順で割り当てる。
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


def _make_pipeline(degree: int) -> Pipeline:
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scale", StandardScaler()),
            ("linear", LinearRegression()),
        ]
    )


def _plot_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    x_col: str,
    y_col: str,
    player_type: str,
    year: int,
    results: list[tuple[int, Pipeline, float, float]],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        X_train[:, 0],
        y_train,
        color=COLOR_MUTED,
        alpha=0.5,
        edgecolor=COLOR_SURFACE,
        linewidth=0.4,
        s=20,
        label="train",
        zorder=2,
    )
    ax.scatter(
        X_test[:, 0],
        y_test,
        color=COLOR_TEST,
        alpha=0.7,
        edgecolor=COLOR_SURFACE,
        linewidth=0.4,
        s=24,
        marker="^",
        label="test",
        zorder=3,
    )

    x_all = np.concatenate([X_train[:, 0], X_test[:, 0]])
    x_min, x_max = x_all.min(), x_all.max()
    x_pad = (x_max - x_min) * 0.05 or 1.0
    x_line = np.linspace(x_min - x_pad, x_max + x_pad, 300).reshape(-1, 1)

    for (degree, model, train_rmse, test_rmse), color in zip(results, CATEGORICAL_COLORS):
        ax.plot(
            x_line[:, 0],
            model.predict(x_line),
            color=color,
            linewidth=1.6,
            label=f"degree={degree} (train RMSE={train_rmse:.2f} / test RMSE={test_rmse:.2f})",
            zorder=4,
        )

    ax.set_title(
        f"{year} {player_type} / {x_col} -> {y_col}: polynomial degree sweep",
        loc="left",
        fontsize=10,
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_axisbelow(True)
    ax.grid(linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


@app.command()
def poly_compare(
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
    degree: list[int] = typer.Option(
        [1, 3, 8], "--degree", help="比較する多項式回帰の次数。複数指定可（--degree 1 --degree 3 ...）"
    ),
    test_size: float = typer.Option(0.2, "--test-size", help="テストデータの割合（0〜1）"),
    random_state: int = typer.Option(42, "--random-state", help="train_test_splitの乱数シード"),
) -> None:
    """指定した年・対象について、x-fieldからy-fieldへの多項式回帰を複数の次数で
    訓練データのみに当てはめ、予測曲線を重ねた図と次数ごとの訓練/テストRMSEの
    表を保存する。次数を上げると訓練RMSEは下がり続けるがテストRMSEはある次数を
    境に反転して上がり始める（過学習）様子を確認できる。

    事前に `data_produce/fetch_leaderboard.py` で `--field x-field --field y-field`
    の組み合わせを取得しておく必要がある。
    """

    if player_type not in ("pitcher", "batter"):
        raise typer.BadParameter("typeは'pitcher'か'batter'を指定してください")
    if x_field == y_field:
        raise typer.BadParameter("x-fieldとy-fieldには異なるフィールドを指定してください")
    if not degree:
        raise typer.BadParameter("degreeは1つ以上指定してください")
    if len(set(degree)) != len(degree):
        raise typer.BadParameter("degreeに重複した値が含まれています")
    if any(d < 1 for d in degree):
        raise typer.BadParameter("degreeは1以上の整数を指定してください")
    if len(degree) > len(CATEGORICAL_COLORS):
        raise typer.BadParameter(f"degreeは{len(CATEGORICAL_COLORS)}個以内で指定してください（配色の都合）")
    if not 0 < test_size < 1:
        raise typer.BadParameter("test_sizeは0〜1の範囲で指定してください")

    logger = setup_logger(Path(__file__).stem)
    _apply_style()

    stat_key = savant_client.stat_key_for([x_field, y_field])
    csv_name = f"leaderboard/{player_type}/{year}/{stat_key}.csv"
    df = read_csv(csv_name)

    x_col, y_col = x_field.value, y_field.value
    data = df[[x_col, y_col]].dropna()
    X = data[[x_col]].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)

    if len(X) * (1 - test_size) <= max(degree):
        raise typer.BadParameter(
            f"訓練データ数（n={len(X) * (1 - test_size):.0f}）が最大のdegree（{max(degree)}）以下です。"
            "degreeを小さくするかデータを増やしてください"
        )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    results: list[tuple[int, Pipeline, float, float]] = []
    rows = []
    for d in degree:
        model = _make_pipeline(d).fit(X_train, y_train)
        train_rmse = root_mean_squared_error(y_train, model.predict(X_train))
        test_rmse = root_mean_squared_error(y_test, model.predict(X_test))
        results.append((d, model, train_rmse, test_rmse))
        rows.append({"degree": d, "train_rmse": train_rmse, "test_rmse": test_rmse})

    logger.info(f"=== [{stat_key}] {csv_name} (n={len(X)}, train={len(X_train)}, test={len(X_test)}) ===")
    for d, _, train_rmse, test_rmse in results:
        note = "（訓練誤差 << テスト誤差＝過学習の兆候）" if test_rmse > train_rmse * 1.5 else ""
        logger.info(f"degree={d}: train RMSE={train_rmse:.3f} / test RMSE={test_rmse:.3f}{note}")

    fig = _plot_comparison(X_train, y_train, X_test, y_test, x_col, y_col, player_type, year, results)
    fig_path = save_figure(fig, f"{player_type}/{year}/{stat_key}_poly_compare.png", dpi=150)
    plt.close(fig)

    table = pd.DataFrame(rows)
    table_path = save_table(table, f"{player_type}/{year}/{stat_key}_poly_compare.csv")

    logger.info(f"図を保存しました: {fig_path}")
    logger.info(f"表を保存しました: {table_path}")


if __name__ == "__main__":
    app()
