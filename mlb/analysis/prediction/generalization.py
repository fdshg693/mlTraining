"""取得済みリーダーボードCSVから、単回帰の汎化性能を可視化するCLI。

`predict.py`は全データで学習・評価しており、「訓練データへの当てはまりの
良さ」と「未知データに対する実際の性能（汎化性能）」を区別していない。
このスクリプトは同じ単回帰モデルについて、以下3種類の評価を並べて示す。

- 全データ学習: `predict.py`と同じく全データで学習・評価したR^2
  （見た目の良さ。汎化性能の指標としては楽観的に出やすい）
- ホールドアウト（乱数シード違いを複数試行）: `train_test_split`で訓練/
  テストに分け、テスト側R^2を乱数シードを変えながら繰り返す。分割の
  運次第でどれだけブレるかを箱ひげ図で見る
- k分割交差検証: `cross_val_score`でk個のfoldそれぞれをテストに使い、
  1回のホールドアウトより安定した汎化性能の見積もりを得る

加えて、単純`train_test_split`と`StratifiedShuffleSplit`（層化サンプリング）
を比較する。mlbの投手データは母数が少なく、層化キーとして選んだカテゴリ
（先発/リリーフの偏り等）が単純分割では運次第で大きくズレやすい。層化キーの
test側カテゴリ比率のズレ・R^2のブレ幅を並べて、「代表性を保つ」効果を確認する。

図中のテキストは英語表記にしている（predict.py等と同様、実行環境にCJK
対応フォントが入っておらず、日本語だとタイトル等が文字化けするため）。
"""

from enum import Enum
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_DIR))
from data_produce import savant_client
from data_produce.savant_client import SavantField
from data_produce.util.file_io import read_csv, save_figure, save_table
from data_produce.util.logging_config import setup_logger


app = typer.Typer(
    help="単回帰の汎化性能（全データ学習/ホールドアウト/交差検証）を比較して可視化するCLI",
    add_completion=False,
)

# dataviz skillの検証済みパレット（ライトモード）からの抜粋。predict.py/role.py等と共通。
COLOR_PRIMARY = "#2a78d6"
COLOR_SECONDARY = "#eb6834"
COLOR_INK = "#0b0b0b"
COLOR_SECONDARY_INK = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"
COLOR_SURFACE = "#fcfcfb"

HOLDOUT = "holdout"
CROSS_VAL = "cross_val"
METHOD_COLORS = {HOLDOUT: COLOR_PRIMARY, CROSS_VAL: COLOR_SECONDARY}

SIMPLE_SPLIT = "simple_split"
STRATIFIED_SPLIT = "stratified_split"
SPLIT_METHOD_COLORS = {SIMPLE_SPLIT: COLOR_PRIMARY, STRATIFIED_SPLIT: COLOR_SECONDARY}

STARTER = "starter"
RELIEVER = "reliever"
STARTER_IP_PER_GAME = 3.0  # role.pyと同じ閾値


class StratifyBy(str, Enum):
    """層化サンプリングに使う層化キーの種類。"""

    TARGET = "target"  # 目的変数をpd.qcutで等頻度カテゴリに分割
    ROLE = "role"  # role.pyと同じ基準の先発/リリーフ区分


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


def _holdout_seed_sweep(X: np.ndarray, y: np.ndarray, test_size: float, seeds: range) -> pd.DataFrame:
    """乱数シードを変えながらホールドアウト評価を繰り返し、testのR^2/MSEを集める。"""

    rows = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        model = LinearRegression().fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append(
            {
                "method": HOLDOUT,
                "run": seed,
                "r2": r2_score(y_test, pred),
                "mse": mean_squared_error(y_test, pred),
            }
        )
    return pd.DataFrame(rows)


def _cross_validate(X: np.ndarray, y: np.ndarray, cv: int) -> pd.DataFrame:
    """k分割交差検証で、foldごとのR^2/MSEを集める。"""

    r2_scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")
    mse_scores = -cross_val_score(LinearRegression(), X, y, cv=cv, scoring="neg_mean_squared_error")
    return pd.DataFrame({"method": CROSS_VAL, "run": range(cv), "r2": r2_scores, "mse": mse_scores})


def _to_decimal_innings(series: pd.Series) -> pd.Series:
    """savantの投球回表記（例: 125.2 = 125回2/3）を実際の小数のイニング数に変換する。

    role.pyと同じ変換（小数点以下はアウト数0/1/2を表す独自表記）。
    """

    whole = np.floor(series)
    outs = np.round((series - whole) * 10)
    return whole + outs / 3


def _role_strata(data: pd.DataFrame, game_col: str, ip_col: str) -> np.ndarray:
    """role.pyと同じ基準（1登板あたり投球回3.0以上で先発）で役割を分類し、層化キーとして返す。"""

    ip = _to_decimal_innings(data[ip_col])
    ip_per_game = ip / data[game_col]
    return np.where(ip_per_game >= STARTER_IP_PER_GAME, STARTER, RELIEVER)


def _target_strata(y: np.ndarray, n_bins: int) -> np.ndarray:
    """目的変数を等頻度のn_binsカテゴリに`pd.qcut`で分け、層化キーとして返す。"""

    return pd.qcut(pd.Series(y), q=n_bins, labels=False, duplicates="drop").to_numpy()


def _split_method_sweep(
    X: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    test_size: float,
    seeds: range,
) -> tuple[pd.DataFrame, pd.Series]:
    """乱数シードを変えながら単純分割/層化分割を繰り返し、test側の層化キー比率のズレとR^2/MSEを集める。

    「比率のズレ」は全体カテゴリ比率とtest側カテゴリ比率の差の絶対値の合計
    （0に近いほどtest側が全体の構成をよく代表している）。
    """

    overall_ratio = pd.Series(strata).value_counts(normalize=True).sort_index()

    def _ratio_deviation(test_strata: np.ndarray) -> float:
        test_ratio = pd.Series(test_strata).value_counts(normalize=True).reindex(overall_ratio.index, fill_value=0.0)
        return float((test_ratio - overall_ratio).abs().sum())

    rows = []
    for seed in seeds:
        X_train, X_test, y_train, y_test, _, strata_test = train_test_split(
            X, y, strata, test_size=test_size, random_state=seed
        )
        pred = LinearRegression().fit(X_train, y_train).predict(X_test)
        rows.append(
            {
                "method": SIMPLE_SPLIT,
                "run": seed,
                "r2": r2_score(y_test, pred),
                "mse": mean_squared_error(y_test, pred),
                "ratio_deviation": _ratio_deviation(strata_test),
            }
        )

    splitter = StratifiedShuffleSplit(n_splits=len(seeds), test_size=test_size, random_state=seeds[0])
    for run, (train_index, test_index) in enumerate(splitter.split(X, strata)):
        pred = LinearRegression().fit(X[train_index], y[train_index]).predict(X[test_index])
        rows.append(
            {
                "method": STRATIFIED_SPLIT,
                "run": run,
                "r2": r2_score(y[test_index], pred),
                "mse": mean_squared_error(y[test_index], pred),
                "ratio_deviation": _ratio_deviation(strata[test_index]),
            }
        )

    return pd.DataFrame(rows), overall_ratio


def _plot_generalization(
    scores: pd.DataFrame,
    r2_all: float,
    x_col: str,
    y_col: str,
    player_type: str,
    year: int,
    test_size: float,
    n_seeds: int,
    cv: int,
) -> plt.Figure:
    groups = [scores.loc[scores["method"] == m, "r2"] for m in (HOLDOUT, CROSS_VAL)]
    labels = [f"holdout\n(test_size={test_size}, n={n_seeds} seeds)", f"{cv}-fold CV\n(n={cv} folds)"]

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
    for patch, method in zip(box["boxes"], (HOLDOUT, CROSS_VAL)):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.35)
        patch.set_edgecolor(METHOD_COLORS[method])

    rng = np.random.default_rng(0)
    for i, (method, g) in enumerate(zip((HOLDOUT, CROSS_VAL), groups), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(g))
        ax.scatter(
            np.full(len(g), i) + jitter,
            g,
            color=METHOD_COLORS[method],
            alpha=0.6,
            edgecolor=COLOR_SURFACE,
            linewidth=0.3,
            s=18,
            zorder=3,
        )

    ax.axhline(
        r2_all,
        color=COLOR_INK,
        linestyle="--",
        linewidth=1.2,
        label=f"all-data fit R2={r2_all:.3f} (optimistic, not out-of-sample)",
        zorder=2,
    )

    ax.set_title(
        f"{year} {player_type} / {x_col} -> {y_col} generalization (holdout vs cross-val)",
        loc="left",
        fontsize=10,
    )
    ax.set_ylabel("R2 (out-of-sample)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def _plot_split_comparison(
    scores: pd.DataFrame,
    overall_ratio: pd.Series,
    x_col: str,
    y_col: str,
    player_type: str,
    year: int,
    test_size: float,
    n_seeds: int,
    stratify_by: StratifyBy,
) -> plt.Figure:
    methods = [SIMPLE_SPLIT, STRATIFIED_SPLIT]
    labels = [f"simple\n(test_size={test_size}, n={n_seeds})", f"stratified\n(test_size={test_size}, n={n_seeds})"]

    fig, (ax_ratio, ax_r2) = plt.subplots(1, 2, figsize=(12, 6))
    panels = (
        (ax_ratio, "ratio_deviation", f"{stratify_by.value} category ratio deviation (test vs overall)"),
        (ax_r2, "r2", "R2 (out-of-sample)"),
    )
    for ax, metric, ylabel in panels:
        groups = [scores.loc[scores["method"] == m, metric] for m in methods]
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
        for patch, method in zip(box["boxes"], methods):
            patch.set_facecolor(SPLIT_METHOD_COLORS[method])
            patch.set_alpha(0.35)
            patch.set_edgecolor(SPLIT_METHOD_COLORS[method])

        rng = np.random.default_rng(0)
        for i, (method, g) in enumerate(zip(methods, groups), start=1):
            jitter = rng.uniform(-0.12, 0.12, size=len(g))
            ax.scatter(
                np.full(len(g), i) + jitter,
                g,
                color=SPLIT_METHOD_COLORS[method],
                alpha=0.6,
                edgecolor=COLOR_SURFACE,
                linewidth=0.3,
                s=18,
                zorder=3,
            )
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    ratio_text = ", ".join(f"{k}={v:.3f}" for k, v in overall_ratio.round(3).items())
    fig.suptitle(
        f"{year} {player_type} / {x_col} -> {y_col} split comparison "
        f"(stratify_by={stratify_by.value}, overall ratio: {ratio_text})",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


@app.command()
def generalization(
    year: int = typer.Option(2026, "--year", "-y", help="対象の年"),
    player_type: str = typer.Option("pitcher", "--type", "-t", help="対象（pitcher または batter）"),
    x_field: SavantField = typer.Option(
        SavantField.FASTBALL_VELOCITY, "--x-field", help="説明変数のフィールド"
    ),
    y_field: SavantField = typer.Option(
        SavantField.FASTBALL_SPIN, "--y-field", help="目的変数のフィールド"
    ),
    test_size: float = typer.Option(0.2, "--test-size", help="ホールドアウトのテスト割合（0〜1）"),
    n_seeds: int = typer.Option(30, "--n-seeds", help="ホールドアウトを繰り返す乱数シードの数"),
    cv: int = typer.Option(5, "--cv", help="交差検証の分割数（k-fold）"),
    random_state: int = typer.Option(42, "--random-state", help="単発ホールドアウト評価に使う乱数シード"),
    stratify_by: StratifyBy = typer.Option(
        StratifyBy.TARGET,
        "--stratify-by",
        help="単純分割 vs 層化分割の比較に使う層化キー（target: 目的変数をpd.qcutで分割 / role: role.pyと同じ先発・リリーフ区分）",
    ),
    n_bins: int = typer.Option(4, "--n-bins", help="stratify-by=targetの場合の層化カテゴリ数（等頻度でpd.qcut）"),
) -> None:
    """単回帰について、全データ学習/ホールドアウト（シード違い複数）/k分割交差検証
    のR^2を比較し、汎化性能のブレと安定化を箱ひげ図で可視化する。

    加えて、単純`train_test_split`と`StratifiedShuffleSplit`（層化サンプリング）を
    比較し、層化キーのtest側カテゴリ比率のズレ・R^2のブレ幅を箱ひげ図で可視化する。

    事前に `data_produce/fetch_leaderboard.py` で `--field x-field --field y-field`
    の組み合わせ（`--stratify-by role`の場合はさらに`--field p_game --field
    p_formatted_ip`も）を取得しておく必要がある。
    """

    if player_type not in ("pitcher", "batter"):
        raise typer.BadParameter("typeは'pitcher'か'batter'を指定してください")
    if x_field == y_field:
        raise typer.BadParameter("x-fieldとy-fieldには異なるフィールドを指定してください")
    if not 0 < test_size < 1:
        raise typer.BadParameter("test_sizeは0〜1の範囲で指定してください")
    if cv < 2:
        raise typer.BadParameter("cvは2以上を指定してください")
    if stratify_by == StratifyBy.TARGET and n_bins < 2:
        raise typer.BadParameter("n_binsは2以上を指定してください")

    logger = setup_logger(Path(__file__).stem)
    _apply_style()

    game_col, ip_col = SavantField.GAMES.value, SavantField.INNINGS_PITCHED.value
    role_fields = [SavantField.GAMES, SavantField.INNINGS_PITCHED] if stratify_by == StratifyBy.ROLE else []
    fields = [x_field, y_field] + [f for f in role_fields if f not in (x_field, y_field)]
    stat_key = savant_client.stat_key_for(fields)
    csv_name = f"leaderboard/{player_type}/{year}/{stat_key}.csv"
    df = read_csv(csv_name)

    x_col, y_col = x_field.value, y_field.value
    cols = [x_col, y_col] + ([game_col, ip_col] if stratify_by == StratifyBy.ROLE else [])
    data = df[cols].dropna().copy()
    X = data[[x_col]].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)

    if len(X) < cv:
        raise typer.BadParameter(f"データ数（n={len(X)}）がcv（{cv}）未満です。cvを小さくするかデータを増やしてください")

    # 全データ学習（predict.pyと同じ評価。見た目の良さの参照値）
    model_all = LinearRegression().fit(X, y)
    r2_all = model_all.score(X, y)

    # 単発ホールドアウト（train/testのMSE・R^2のズレを確認）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model_holdout = LinearRegression().fit(X_train, y_train)
    train_pred = model_holdout.predict(X_train)
    test_pred = model_holdout.predict(X_test)
    train_r2, train_mse = r2_score(y_train, train_pred), mean_squared_error(y_train, train_pred)
    test_r2, test_mse = r2_score(y_test, test_pred), mean_squared_error(y_test, test_pred)

    # ホールドアウトのシード違い複数試行 + k分割交差検証
    seed_sweep = _holdout_seed_sweep(X, y, test_size, range(n_seeds))
    cv_scores = _cross_validate(X, y, cv)
    scores = pd.concat([seed_sweep, cv_scores], ignore_index=True)

    logger.info(f"=== [{stat_key}] {csv_name} (n={len(X)}) ===")
    logger.info(f"全データ学習: R^2={r2_all:.3f}（見た目の良さ。汎化性能の指標としては楽観的に出やすい）")
    logger.info(
        f"単発ホールドアウト(random_state={random_state}, test_size={test_size}): "
        f"train R^2={train_r2:.3f} MSE={train_mse:.3f} / test R^2={test_r2:.3f} MSE={test_mse:.3f}"
    )
    logger.info(
        f"ホールドアウト{n_seeds}シード: R^2 平均={seed_sweep['r2'].mean():.3f} "
        f"標準偏差={seed_sweep['r2'].std():.3f} min={seed_sweep['r2'].min():.3f} max={seed_sweep['r2'].max():.3f}"
    )
    logger.info(
        f"{cv}-fold交差検証: R^2 各fold={[f'{s:.3f}' for s in cv_scores['r2']]} "
        f"平均={cv_scores['r2'].mean():.3f} 標準偏差={cv_scores['r2'].std():.3f}"
    )

    fig = _plot_generalization(scores, r2_all, x_col, y_col, player_type, year, test_size, n_seeds, cv)
    fig_path = save_figure(fig, f"{player_type}/{year}/{stat_key}_generalization.png", dpi=150)
    plt.close(fig)

    table_path = save_table(scores, f"{player_type}/{year}/{stat_key}_generalization.csv")

    logger.info(f"図を保存しました: {fig_path}")
    logger.info(f"表を保存しました: {table_path}")

    # 単純分割 vs 層化分割（代表性を保った分割）の比較
    if stratify_by == StratifyBy.TARGET:
        strata = _target_strata(y, n_bins)
    else:
        strata = _role_strata(data, game_col, ip_col)

    strata_counts = pd.Series(strata).value_counts()
    if strata_counts.min() < 2:
        raise typer.BadParameter(
            f"層化キー（{stratify_by.value}）の最小カテゴリのサンプル数が{strata_counts.min()}件と少なすぎます。"
            "stratify-by=targetならn-binsを減らす、stratify-by=roleなら対象データを見直してください"
        )

    split_scores, overall_ratio = _split_method_sweep(X, y, strata, test_size, range(n_seeds))
    simple_scores = split_scores.loc[split_scores["method"] == SIMPLE_SPLIT]
    stratified_scores = split_scores.loc[split_scores["method"] == STRATIFIED_SPLIT]

    logger.info(f"=== 単純分割 vs 層化分割（層化キー: {stratify_by.value}） ===")
    logger.info(f"全体のカテゴリ比率: {overall_ratio.round(3).to_dict()}")
    logger.info(
        f"単純分割: testカテゴリ比率ズレ 平均={simple_scores['ratio_deviation'].mean():.3f} "
        f"標準偏差={simple_scores['ratio_deviation'].std():.3f} / "
        f"R^2 平均={simple_scores['r2'].mean():.3f} 標準偏差={simple_scores['r2'].std():.3f}"
    )
    logger.info(
        f"層化分割: testカテゴリ比率ズレ 平均={stratified_scores['ratio_deviation'].mean():.3f} "
        f"標準偏差={stratified_scores['ratio_deviation'].std():.3f} / "
        f"R^2 平均={stratified_scores['r2'].mean():.3f} 標準偏差={stratified_scores['r2'].std():.3f}"
    )

    split_fig = _plot_split_comparison(
        split_scores, overall_ratio, x_col, y_col, player_type, year, test_size, n_seeds, stratify_by
    )
    split_fig_path = save_figure(
        split_fig, f"{player_type}/{year}/{stat_key}_split_comparison_{stratify_by.value}.png", dpi=150
    )
    plt.close(split_fig)

    split_table_path = save_table(
        split_scores, f"{player_type}/{year}/{stat_key}_split_comparison_{stratify_by.value}.csv"
    )

    logger.info(f"分割比較の図を保存しました: {split_fig_path}")
    logger.info(f"分割比較の表を保存しました: {split_table_path}")


if __name__ == "__main__":
    app()
