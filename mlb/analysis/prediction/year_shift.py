"""複数年のリーダーボードCSVから、年をまたいだ汎化（分布シフト）を確認するCLI。

`compare_models.py`は単一年のデータをtrain/testに分けて評価しており、
`generalization.py`のk分割交差検証も同一年内でのfold分割にすぎない。
どちらも「同じ年（同じ分布）の中でどれだけ汎化するか」しか測れておらず、
「来シーズンのデータに対してどれだけ通用するか」は分からない。

このスクリプトは`compare_models.py`と同じ特徴量セット・ERA予測について、
- train-dev: 複数年（例: 2023-2024年）のデータをまとめて学習し、
  その中でのk分割交差検証RMSE（同一分布内の汎化性能の見積もり）
- year-shift test: train-devでは一度も見ていない別の年（例: 2025年）を
  テストとして評価したRMSE（未知の分布＝年への汎化性能）

を並べて示し、両者の差（分布シフトのコスト）を確認する。

事前に`data_produce/fetch_leaderboard.py`でtrain-dev年・test年それぞれの
CSVを取得しておく必要がある（`--year`を変えて複数回実行）。

図中のテキストは英語表記にしている（compare_models.py等と同様、実行環境に
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_DIR))
from data_produce import savant_client
from data_produce.savant_client import SavantField
from data_produce.util.file_io import read_csv, save_figure, save_table
from data_produce.util.logging_config import setup_logger


app = typer.Typer(
    help="複数年をtrain-devにまとめて学習し、別の年をtestとして評価することで、"
    "年をまたいだ分布シフトへの汎化性能を確認するCLI",
    add_completion=False,
)

DEFAULT_FEATURES = [
    SavantField.FASTBALL_VELOCITY,
    SavantField.FASTBALL_SPIN,
    SavantField.INNINGS_PITCHED,
    SavantField.GAMES,
    SavantField.FASTBALL_USAGE,
    SavantField.SINKER_USAGE,
    SavantField.SLIDER_USAGE,
    SavantField.CURVEBALL_USAGE,
    SavantField.CHANGEUP_USAGE,
    SavantField.CUTTER_USAGE,
]

LINEAR_REGRESSION = "linear_regression"
RANDOM_FOREST = "random_forest"

# dataviz skillの検証済みパレット（ライトモード）からの抜粋。compare_models.py/generalization.pyと共通。
COLOR_PRIMARY = "#2a78d6"
COLOR_SECONDARY = "#eb6834"
COLOR_INK = "#0b0b0b"
COLOR_SECONDARY_INK = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"
COLOR_SURFACE = "#fcfcfb"
MODEL_COLORS = {LINEAR_REGRESSION: COLOR_PRIMARY, RANDOM_FOREST: COLOR_SECONDARY}


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


def build_models(feature_cols: list[str], random_state: int) -> dict[str, Pipeline]:
    """数値特徴量の標準化（ColumnTransformer）と各回帰器を組み合わせたパイプラインを作る。

    `compare_models.py`と同じ構成（線形回帰 / random_forest）。
    """

    def preprocessing() -> ColumnTransformer:
        return ColumnTransformer([("num", StandardScaler(), feature_cols)])

    return {
        LINEAR_REGRESSION: Pipeline([("preprocessing", preprocessing()), ("model", LinearRegression())]),
        RANDOM_FOREST: Pipeline(
            [("preprocessing", preprocessing()), ("model", RandomForestRegressor(random_state=random_state))]
        ),
    }


def _load_year(
    player_type: str, year: int, stat_key: str, feature_cols: list[str], target_col: str
) -> pd.DataFrame:
    """1年分のリーダーボードCSVを読み込み、特徴量+目的変数の欠損処理を行って返す。"""

    csv_name = f"leaderboard/{player_type}/{year}/{stat_key}.csv"
    df = read_csv(csv_name)
    data = df[feature_cols + [target_col]].copy()
    # 球種使用率（n_*_formatted）のNaNは「その球種を投げていない」＝0%であり欠損ではないため、
    # dropnaで行ごと落とす前に0埋めする（compare_models.pyと同じ扱い。data_produce/data/NOTES.md参照）。
    usage_cols = [col for col in feature_cols if col.startswith("n_") and col.endswith("_formatted")]
    data[usage_cols] = data[usage_cols].fillna(0)
    data = data.dropna()
    data["year"] = year
    return data


def cross_val_rmse(model: Pipeline, X: pd.DataFrame, y: pd.Series, cv: int) -> np.ndarray:
    """cv分割の交差検証RMSEを配列で返す（scoringの符号を反転させる）。"""

    neg_rmses = cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=cv)
    return -neg_rmses


def _plot_year_shift(
    cv_scores: pd.DataFrame,
    shift_rmse: dict[str, float],
    target_col: str,
    player_type: str,
    train_years: list[int],
    test_year: int,
    cv: int,
) -> plt.Figure:
    model_names = [LINEAR_REGRESSION, RANDOM_FOREST]
    groups = [cv_scores.loc[cv_scores["model"] == name, "rmse"] for name in model_names]
    train_years_label = "+".join(str(y) for y in train_years)

    fig, ax = plt.subplots(figsize=(7, 6))
    box = ax.boxplot(
        groups,
        tick_labels=model_names,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": COLOR_INK, "linewidth": 1.5},
        whiskerprops={"color": COLOR_MUTED},
        capprops={"color": COLOR_MUTED},
        flierprops={"markeredgecolor": COLOR_MUTED, "markersize": 4},
    )
    for patch, name in zip(box["boxes"], model_names):
        patch.set_facecolor(MODEL_COLORS[name])
        patch.set_alpha(0.35)
        patch.set_edgecolor(MODEL_COLORS[name])

    rng = np.random.default_rng(0)
    for i, (name, g) in enumerate(zip(model_names, groups), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(g))
        ax.scatter(
            np.full(len(g), i) + jitter,
            g,
            color=MODEL_COLORS[name],
            alpha=0.6,
            edgecolor=COLOR_SURFACE,
            linewidth=0.3,
            s=18,
            zorder=3,
        )

    for name in model_names:
        ax.axhline(
            shift_rmse[name],
            color=MODEL_COLORS[name],
            linestyle="--",
            linewidth=1.5,
            label=f"{name} year-shift test ({test_year}) RMSE={shift_rmse[name]:.3f}",
            zorder=2,
        )

    ax.set_title(
        f"{player_type} / predicting {target_col}: train-dev({train_years_label}) {cv}-fold CV "
        f"vs year-shift test({test_year})",
        loc="left",
        fontsize=10,
    )
    ax.set_ylabel("RMSE")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


@app.command()
def year_shift(
    train_year: list[int] = typer.Option(
        [2023, 2024], "--train-year", help="train-devに使う年。複数指定可（--train-year 2023 --train-year 2024）"
    ),
    test_year: int = typer.Option(2025, "--test-year", help="year-shiftのテストに使う年（train-devとは別の年）"),
    player_type: str = typer.Option("pitcher", "--type", "-t", help="対象（pitcher または batter）"),
    feature_field: list[SavantField] = typer.Option(
        DEFAULT_FEATURES, "--feature", "-f", help="説明変数のフィールド。複数指定可（-f ff_avg_speed -f p_game ...）"
    ),
    target_field: SavantField = typer.Option(SavantField.ERA, "--target-field", help="目的変数のフィールド"),
    cv: int = typer.Option(5, "--cv", help="train-dev内でのcross_val_score分割数"),
    random_state: int = typer.Option(42, "--random-state", help="random_forestに使う乱数シード"),
) -> None:
    """`compare_models.py`と同じ特徴量セットで、複数年（train-dev）をまとめて学習した
    モデルを、train-dev内のk分割交差検証RMSEと、未知の年（test）に対するRMSEとで比較する。

    事前に `data_produce/fetch_leaderboard.py` で train-dev・test それぞれの年について
    `--feature`（feature分すべて）と `--target-field` を組み合わせて取得しておく必要がある。
    """

    if player_type not in ("pitcher", "batter"):
        raise typer.BadParameter("typeは'pitcher'か'batter'を指定してください")
    if target_field in feature_field:
        raise typer.BadParameter("target-fieldはfeatureに含めないでください")
    if len(set(feature_field)) != len(feature_field):
        raise typer.BadParameter("featureに重複したフィールドが含まれています")
    if len(feature_field) < 2:
        raise typer.BadParameter("featureは2つ以上指定してください（多変量回帰のため）")
    if len(set(train_year)) != len(train_year):
        raise typer.BadParameter("train-yearに重複した年が含まれています")
    if test_year in train_year:
        raise typer.BadParameter("test-yearはtrain-yearに含まれない年を指定してください")
    if cv < 2:
        raise typer.BadParameter("cvは2以上を指定してください")

    logger = setup_logger(Path(__file__).stem)
    _apply_style()

    train_years = sorted(train_year)
    stat_key = savant_client.stat_key_for(list(feature_field) + [target_field])
    feature_cols = [field.value for field in feature_field]
    target_col = target_field.value

    train_dev = pd.concat(
        [_load_year(player_type, year, stat_key, feature_cols, target_col) for year in train_years],
        ignore_index=True,
    )
    test_data = _load_year(player_type, test_year, stat_key, feature_cols, target_col)

    if len(train_dev) <= cv:
        raise typer.BadParameter(f"train-devのデータ数（n={len(train_dev)}）がcv（{cv}）以下です")
    if len(test_data) == 0:
        raise typer.BadParameter(f"test-year（{test_year}）のデータが0件です")

    X_train_dev, y_train_dev = train_dev[feature_cols], train_dev[target_col]
    X_test, y_test = test_data[feature_cols], test_data[target_col]

    train_years_label = "+".join(str(y) for y in train_years)
    logger.info(f"=== [{stat_key}] train-dev={train_years_label} (n={len(train_dev)}) / test={test_year} (n={len(test_data)}) ===")
    logger.info(f"特徴量({len(feature_cols)}件): {feature_cols}")
    logger.info(f"目的変数: {target_col}")
    for year in train_years:
        logger.info(f"  train-dev {year}: n={(train_dev['year'] == year).sum()}")

    models = build_models(feature_cols, random_state)

    logger.info(f"\n=== train-dev内の{cv}-fold cross_val_score（同一分布内の汎化性能の見積もり） ===")
    cv_rows = []
    for name, model in models.items():
        rmses = cross_val_rmse(model, X_train_dev, y_train_dev, cv)
        cv_rows += [{"model": name, "fold": fold, "rmse": rmse} for fold, rmse in enumerate(rmses)]
        logger.info(f"{name}: mean={rmses.mean():.4f}  std={rmses.std():.4f}  min={rmses.min():.4f}  max={rmses.max():.4f}")
    cv_scores = pd.DataFrame(cv_rows)

    logger.info(f"\n=== year-shift test（train-dev全体で学習 → 未知の年{test_year}で評価） ===")
    summary_rows = []
    shift_rmse = {}
    for name, model in models.items():
        model.fit(X_train_dev, y_train_dev)
        pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, pred)
        shift_rmse[name] = rmse
        cv_mean = cv_scores.loc[cv_scores["model"] == name, "rmse"].mean()
        cv_std = cv_scores.loc[cv_scores["model"] == name, "rmse"].std()
        gap = rmse - cv_mean
        summary_rows.append(
            {
                "model": name,
                "cv_mean_rmse": cv_mean,
                "cv_std_rmse": cv_std,
                "year_shift_rmse": rmse,
                "gap": gap,
            }
        )
        logger.info(
            f"{name}: year-shift RMSE={rmse:.4f}（train-dev CV平均={cv_mean:.4f}, 差={gap:+.4f}）"
        )
    summary = pd.DataFrame(summary_rows)

    fig = _plot_year_shift(cv_scores, shift_rmse, target_col, player_type, train_years, test_year, cv)
    run_id = f"train{train_years_label}_test{test_year}"
    fig_path = save_figure(fig, f"{player_type}/{stat_key}_{run_id}_year_shift.png", dpi=150)
    plt.close(fig)

    cv_table_path = save_table(cv_scores, f"{player_type}/{stat_key}_{run_id}_year_shift_cv_scores.csv")
    summary_table_path = save_table(summary, f"{player_type}/{stat_key}_{run_id}_year_shift_summary.csv")

    logger.info(f"図を保存しました: {fig_path}")
    logger.info(f"表（CVスコア）を保存しました: {cv_table_path}")
    logger.info(f"表（サマリ）を保存しました: {summary_table_path}")


if __name__ == "__main__":
    app()
