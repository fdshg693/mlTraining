"""取得済みリーダーボードCSVから、投手の球種構成・球速・回転数を特徴量に
KMeansでタイプ分け（クラスタリング）するCLI。

`role/role.py`はイニング数の閾値で先発/リリーフを機械的に分類しているだけで、
「似た投手」をデータから発見する教師なし学習は未実施だった。このスクリプトは
主要球種の使用率・4シーム球速・回転数（`CLUSTER_FIELDS`）を`StandardScaler`で
標準化し`KMeans`でクラスタリングする。kは
`handson-ml3/01/learn/elbow_silhouette.py`と同様にinertia（エルボー法）と
silhouette_scoreの両方を見て選び（デフォルトはsilhouette_score最大のkを自動選択）、
`PCA`で2次元に落とした散布図で可視化する。role.pyと同じ基準（1登板あたり投球回
3.0以上で先発）で求めた先発/リリーフ区分も点の形で重ねて表示し、クラスタが
役割と一致するか・しないかを比較できるようにする。

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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_DIR))
from data_produce import savant_client
from data_produce.savant_client import SavantField
from data_produce.util.file_io import read_csv, save_figure, save_table
from data_produce.util.logging_config import setup_logger


app = typer.Typer(help="投手を球種構成・球速・回転数でクラスタリングするCLI", add_completion=False)

# クラスタリングに使う特徴量: 主要球種の使用率 + 4シームの球速・回転数。
# role.pyの先発/リリーフ判定に使うp_game/p_formatted_ipはここには含めない
# （クラスタリング後に役割との一致・不一致を比較する軸として別に使うため）。
USAGE_FIELDS = [
    SavantField.FASTBALL_USAGE,
    SavantField.SINKER_USAGE,
    SavantField.SLIDER_USAGE,
    SavantField.CURVEBALL_USAGE,
    SavantField.CHANGEUP_USAGE,
    SavantField.CUTTER_USAGE,
]
CLUSTER_FIELDS = [*USAGE_FIELDS, SavantField.FASTBALL_VELOCITY, SavantField.FASTBALL_SPIN]

STARTER = "starter"
RELIEVER = "reliever"
STARTER_IP_PER_GAME = 3.0  # role.pyと同じ閾値

# dataviz skillの検証済みパレット（ライトモード）からの抜粋。role.py等と共通。
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
ROLE_MARKERS = {STARTER: "o", RELIEVER: "^"}


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


def _to_decimal_innings(series: pd.Series) -> pd.Series:
    """savantの投球回表記（例: 125.2 = 125回2/3）を実際の小数のイニング数に変換する。

    小数点以下は10進の端数ではなく、アウト数（0/1/2）を表す独自表記のため。
    """

    whole = np.floor(series)
    outs = np.round((series - whole) * 10)
    return whole + outs / 3


def _classify_role(data: pd.DataFrame, game_col: str, ip_col: str) -> np.ndarray:
    """role.pyと同じ基準（1登板あたり投球回3.0以上で先発）で役割を分類する。"""

    ip = _to_decimal_innings(data[ip_col])
    ip_per_game = ip / data[game_col]
    return np.where(ip_per_game >= STARTER_IP_PER_GAME, STARTER, RELIEVER)


def _select_k(X: np.ndarray, k_min: int, k_max: int, random_state: int) -> pd.DataFrame:
    """kをk_min〜k_maxで振り、inertia（エルボー法）とsilhouette_scoreを比較する。"""

    rows = []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(X)
        rows.append({"k": k, "inertia": kmeans.inertia_, "silhouette_score": silhouette_score(X, labels)})
    return pd.DataFrame(rows)


def _plot_k_selection(scores: pd.DataFrame, best_k: int) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(scores["k"], scores["inertia"], "o-", color=COLOR_INK)
    ax1.set_xlabel("k")
    ax1.set_ylabel("inertia")
    ax1.set_title("Elbow method", loc="left")

    ax2.plot(scores["k"], scores["silhouette_score"], "o-", color=COLOR_INK)
    ax2.axvline(best_k, color=CATEGORICAL_COLORS[1], linestyle="--", label=f"best k={best_k}")
    ax2.set_xlabel("k")
    ax2.set_ylabel("silhouette score")
    ax2.set_title("Silhouette score", loc="left")
    ax2.legend(frameon=False)

    for ax in (ax1, ax2):
        ax.set_axisbelow(True)
        ax.grid(linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def _plot_clusters(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    role_labels: np.ndarray,
    explained_variance: np.ndarray,
    year: int,
    k: int,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 7))
    for cluster_id in sorted(set(cluster_labels)):
        cluster_mask = cluster_labels == cluster_id
        color = CATEGORICAL_COLORS[int(cluster_id) % len(CATEGORICAL_COLORS)]
        for role, marker in ROLE_MARKERS.items():
            mask = cluster_mask & (role_labels == role)
            if not mask.any():
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                color=color,
                marker=marker,
                alpha=0.7,
                edgecolor=COLOR_SURFACE,
                linewidth=0.4,
                s=30,
                label=f"cluster {cluster_id} / {role}",
            )

    ax.set_title(f"{year} pitcher clusters (k={k}) in PCA space vs role", loc="left", fontsize=10)
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
def cluster(
    year: int = typer.Option(2026, "--year", "-y", help="対象の年"),
    k: int = typer.Option(
        0, "--k", help="クラスタ数。0（デフォルト）ならk-min〜k-maxをsilhouette_scoreで自動選定する"
    ),
    k_min: int = typer.Option(2, "--k-min", help="自動選定時に試すkの最小値"),
    k_max: int = typer.Option(8, "--k-max", help="自動選定時に試すkの最大値"),
    random_state: int = typer.Option(42, "--random-state", help="KMeans/PCAの乱数シード"),
) -> None:
    """投手を球種構成・球速・回転数でクラスタリングし、PCA平面上に可視化する。

    事前に`data_produce/fetch_leaderboard.py`で`p_game`・`p_formatted_ip`と
    `CLUSTER_FIELDS`（球種別使用率6種+`ff_avg_speed`+`ff_avg_spin`）の組み合わせを
    取得しておく必要がある。「q」（規定投球回）フィルタだとリリーフ投手の大半が
    除外されるため、role.py同様`--min-sample`に十分小さい値（例: `40`）を指定すること。

    ```
    uv run python mlb/data_produce/fetch_leaderboard.py --year 2026 --type pitcher \\
      --field p_game --field p_formatted_ip \\
      --field n_ff_formatted --field n_si_formatted --field n_sl_formatted \\
      --field n_cu_formatted --field n_ch_formatted --field n_fc_formatted \\
      --field ff_avg_speed --field ff_avg_spin --min-sample 40
    ```
    """

    if k_min < 2:
        raise typer.BadParameter("k-minは2以上を指定してください")
    if k_max < k_min:
        raise typer.BadParameter("k-maxはk-min以上を指定してください")
    if k != 0 and k < 2:
        raise typer.BadParameter("kは2以上を指定してください（0で自動選定）")

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
    data = df[[name_col, game_col, ip_col, *feature_cols]].copy()
    # 投げていない球種は使用率が0%ではなく欠測(NaN)で返るため、使用率だけ0で補完してから
    # 残りの欠測（フォーシーム自体を投げない投手など）を落とす。dropnaを先にかけると
    # 「全球種を投げる投手」しか残らず母集団が大きく偏ってしまう。
    usage_cols = [field.value for field in USAGE_FIELDS]
    data[usage_cols] = data[usage_cols].fillna(0.0)
    data = data.dropna().reset_index(drop=True)
    role_labels = _classify_role(data, game_col, ip_col)

    if k != 0 and len(data) <= k:
        raise typer.BadParameter(f"データ数（n={len(data)}）がk（{k}）以下です。kを小さくするかデータを増やしてください")
    if k == 0 and len(data) <= k_max:
        raise typer.BadParameter(f"データ数（n={len(data)}）がk-max（{k_max}）以下です。k-maxを小さくするかデータを増やしてください")

    X = StandardScaler().fit_transform(data[feature_cols].to_numpy(dtype=float))

    logger.info(f"=== [{stat_key}] {csv_name} (n={len(X)}) ===")

    if k == 0:
        scores = _select_k(X, k_min, k_max, random_state)
        for _, row in scores.iterrows():
            logger.info(f"k={int(row['k'])}: inertia={row['inertia']:.1f}, silhouette_score={row['silhouette_score']:.3f}")
        k = int(scores.loc[scores["silhouette_score"].idxmax(), "k"])
        logger.info(f"silhouette_score最大のk: {k}")

        k_fig = _plot_k_selection(scores, k)
        k_fig_path = save_figure(k_fig, f"pitcher/{year}/{stat_key}_k_selection.png", dpi=150)
        plt.close(k_fig)
        logger.info(f"k選定図を保存しました: {k_fig_path}")
    else:
        logger.info(f"kを{k}に固定します（自動選定はスキップ）")

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X)
    data["cluster"] = cluster_labels
    data["role"] = role_labels

    crosstab = pd.crosstab(data["cluster"], data["role"])
    logger.info(f"クラスタ x 役割のクロス集計:\n{crosstab}")

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X)

    fig = _plot_clusters(coords, cluster_labels, role_labels, pca.explained_variance_ratio_, year, k)
    fig_path = save_figure(fig, f"pitcher/{year}/{stat_key}_clusters.png", dpi=150)
    plt.close(fig)

    table_path = save_table(data, f"pitcher/{year}/{stat_key}_clusters.csv")
    crosstab_path = save_table(crosstab.reset_index(), f"pitcher/{year}/{stat_key}_cluster_role_crosstab.csv")

    logger.info(f"図を保存しました: {fig_path}")
    logger.info(f"クラスタ付きデータを保存しました: {table_path}")
    logger.info(f"クロス集計を保存しました: {crosstab_path}")


if __name__ == "__main__":
    app()
