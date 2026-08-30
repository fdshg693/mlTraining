"""取得済みリーダーボードCSVから、DBSCANで「典型的でない投手（外れ値）」を検出するCLI。

`cluster.py`のKMeansは全投手を必ずどこかのクラスタへ割り当てるため、
「どのタイプにも当てはまらない投手」を見つける用途には向かない。このスクリプトは
`cluster.py`と同じ特徴量（球種別使用率6種+4シーム球速+回転数、`CLUSTER_FIELDS`）を
`StandardScaler`で標準化し`DBSCAN`を適用する。DBSCANはクラスタ数を指定せず密度で
クラスタを発見し、どのクラスタのコア点の近傍（`eps`以内）にも属さなかった点を
「ノイズ（label=-1）」として外れ値扱いする
（`handson-ml3/01/learn/dbscan_clustering.py`と同じ考え方）。

`eps`・`min_samples`は自動選定の標準的な指標が無い決め打ちパラメータで、
小さすぎるとほぼ全点がノイズになり、大きすぎると1クラスタに潰れる。デフォルト値は
本データセットで試した一例に過ぎないため、まずログに出る「クラスタ数・ノイズ点数」を
見ながら値を調整すること。

図中のテキストは英語表記にしている（他のanalysisスクリプトと同様、実行環境に
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
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_DIR))
from data_produce import savant_client
from data_produce.savant_client import SavantField
from data_produce.util.file_io import read_csv, save_figure, save_table
from data_produce.util.logging_config import setup_logger


app = typer.Typer(help="DBSCANで典型的でない投手（外れ値）を検出するCLI", add_completion=False)

# cluster.pyと同じ特徴量（同じCSVをそのまま使い回せるようにする）
USAGE_FIELDS = [
    SavantField.FASTBALL_USAGE,
    SavantField.SINKER_USAGE,
    SavantField.SLIDER_USAGE,
    SavantField.CURVEBALL_USAGE,
    SavantField.CHANGEUP_USAGE,
    SavantField.CUTTER_USAGE,
]
CLUSTER_FIELDS = [*USAGE_FIELDS, SavantField.FASTBALL_VELOCITY, SavantField.FASTBALL_SPIN]

NOISE_LABEL = -1

# dataviz skillの検証済みパレット（ライトモード）からの抜粋。cluster.py等と共通。
COLOR_INK = "#0b0b0b"
COLOR_SECONDARY_INK = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"
COLOR_SURFACE = "#fcfcfb"
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
NOISE_COLOR = COLOR_INK


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


def _plot_outliers(
    coords: np.ndarray,
    labels: np.ndarray,
    names: pd.Series,
    explained_variance: np.ndarray,
    year: int,
    eps: float,
    min_samples: int,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 7))

    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        is_noise = cluster_id == NOISE_LABEL
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            color=NOISE_COLOR if is_noise else CATEGORICAL_COLORS[int(cluster_id) % len(CATEGORICAL_COLORS)],
            marker="x" if is_noise else "o",
            alpha=0.9 if is_noise else 0.6,
            edgecolor=None if is_noise else COLOR_SURFACE,
            linewidth=1.4 if is_noise else 0.4,
            s=40 if is_noise else 26,
            label=f"noise (n={mask.sum()})" if is_noise else f"cluster {cluster_id} (n={mask.sum()})",
            zorder=4 if is_noise else 3,
        )

    for name, (x, y) in zip(names[labels == NOISE_LABEL], coords[labels == NOISE_LABEL]):
        ax.annotate(name, xy=(x, y), fontsize=7, color=COLOR_SECONDARY_INK, xytext=(4, 4), textcoords="offset points")

    ax.set_title(
        f"{year} pitcher outliers via DBSCAN (eps={eps}, min_samples={min_samples})", loc="left", fontsize=10
    )
    ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({explained_variance[1]:.1%} var)")
    ax.set_axisbelow(True)
    ax.grid(linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


@app.command()
def outliers(
    year: int = typer.Option(2026, "--year", "-y", help="対象の年"),
    eps: float = typer.Option(2.0, "--eps", help="DBSCANのeps（近傍とみなす標準化後の最大距離）"),
    min_samples: int = typer.Option(5, "--min-samples", help="DBSCANのmin_samples（コア点とみなす近傍点数）"),
) -> None:
    """投手を球種構成・球速・回転数でDBSCANにかけ、どのクラスタにも属さない
    「典型的でない投手」をPCA平面上に可視化する。

    事前に`cluster.py`と同じ`data_produce/fetch_leaderboard.py`コマンド
    （`--field p_game --field p_formatted_ip`+球種別使用率6種+`ff_avg_speed`+
    `ff_avg_spin`、`--min-sample 40`程度）でCSVを取得しておく必要がある。

    `eps`/`min_samples`はデータセットに依存する決め打ちパラメータで自動選定はできない。
    小さすぎるとほぼ全点がノイズになり、大きすぎるとノイズが検出されなくなる。
    まずデフォルト値で実行し、ログのクラスタ数・ノイズ点数を見ながら調整すること。
    """

    if eps <= 0:
        raise typer.BadParameter("epsは正の値を指定してください")
    if min_samples < 1:
        raise typer.BadParameter("min_samplesは1以上を指定してください")

    logger = setup_logger(Path(__file__).stem)
    _apply_style()

    game_col = SavantField.GAMES.value
    ip_col = SavantField.INNINGS_PITCHED.value
    feature_cols = [field.value for field in CLUSTER_FIELDS]

    fields = [SavantField.GAMES, SavantField.INNINGS_PITCHED, *CLUSTER_FIELDS]
    stat_key = savant_client.stat_key_for(fields)
    csv_name = f"leaderboard/pitcher/{year}/{stat_key}.csv"
    df = read_csv(csv_name)

    name_col = "last_name, first_name"
    data = df[[name_col, *feature_cols]].copy()
    # cluster.pyと同様、投げていない球種は使用率が0%ではなく欠測(NaN)で返るため
    # 使用率だけ0で補完してから残りの欠測を落とす（詳細はcluster.pyのコメント参照）。
    usage_cols = [field.value for field in USAGE_FIELDS]
    data[usage_cols] = data[usage_cols].fillna(0.0)
    data = data.dropna().reset_index(drop=True)

    X = StandardScaler().fit_transform(data[feature_cols].to_numpy(dtype=float))

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    data["dbscan_label"] = labels

    n_clusters = len(set(labels)) - (1 if NOISE_LABEL in labels else 0)
    n_noise = int((labels == NOISE_LABEL).sum())

    logger.info(f"=== [{stat_key}] {csv_name} (n={len(X)}) ===")
    logger.info(f"eps={eps}, min_samples={min_samples}")
    logger.info(f"クラスタ数: {n_clusters}, ノイズ点数（外れ値扱い）: {n_noise}")
    if n_noise:
        noise_names = data.loc[labels == NOISE_LABEL, name_col].tolist()
        logger.info(f"典型的でない投手: {noise_names}")

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    fig = _plot_outliers(coords, labels, data[name_col], pca.explained_variance_ratio_, year, eps, min_samples)
    fig_path = save_figure(fig, f"pitcher/{year}/{stat_key}_outliers.png", dpi=150)
    plt.close(fig)

    table_path = save_table(data, f"pitcher/{year}/{stat_key}_outliers.csv")

    logger.info(f"図を保存しました: {fig_path}")
    logger.info(f"DBSCANラベル付きデータを保存しました: {table_path}")


if __name__ == "__main__":
    app()
