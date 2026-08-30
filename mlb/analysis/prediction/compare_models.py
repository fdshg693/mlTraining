"""取得済みリーダーボードCSVから、複数特徴量による回帰モデルを比較・チューニングするCLI。

`predict.py`/`knn_compare.py`/`generalization.py`は常に1x→1yの単回帰だが、
このスクリプトは複数の投球フィールド（球速・回転数・球種使用率・イニング数・
登板数など）を特徴量にして、打者成績側の1フィールド（防御率・被打率など）を
目的変数とする多変量回帰を扱う。

- `ColumnTransformer`で数値特徴量の標準化をまとめ、線形回帰と
  `RandomForestRegressor`を`cross_val_score`（RMSE）で比較する
- `random_forest`に対して`RandomizedSearchCV`でハイパーパラメータ
  （`n_estimators`/`max_depth`/`max_features`/`min_samples_leaf`）を探索する
- 探索後のモデルの特徴量重要度（不純度ベース、MDI）を確認し、
  `sklearn.inspection.permutation_importance`（ホールドアウトでの性能低下ベース）
  と比較する。さらに`PartialDependenceDisplay`で上位特徴量と予測値の関係を確認する
- `joblib`でモデルを保存・再読込して予測が一致することを確認する

図中のテキストは英語表記にしている（predict.py等と同様、実行環境にCJK対応
フォントが入っておらず、日本語だとタイトル等が文字化けするため）。
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_DIR))
from data_produce import savant_client
from data_produce.savant_client import SavantField
from data_produce.util.file_io import output_path, read_csv, save_figure, save_table
from data_produce.util.logging_config import setup_logger
from data_produce.util.metadata import record_read, record_write


app = typer.Typer(
    help="複数特徴量の回帰モデル（線形回帰/random_forest）をcross_val_scoreで比較し、"
    "RandomizedSearchCVでハイパーパラメータ探索・特徴量重要度・モデル永続化まで行うCLI",
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

# dataviz skillの検証済みパレット（ライトモード）からの抜粋。predict.py/generalization.pyと共通。
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
    """数値特徴量の標準化（ColumnTransformer）と各回帰器を組み合わせたパイプラインを作る。"""

    def preprocessing() -> ColumnTransformer:
        return ColumnTransformer([("num", StandardScaler(), feature_cols)])

    return {
        LINEAR_REGRESSION: Pipeline([("preprocessing", preprocessing()), ("model", LinearRegression())]),
        RANDOM_FOREST: Pipeline(
            [("preprocessing", preprocessing()), ("model", RandomForestRegressor(random_state=random_state))]
        ),
    }


def cross_val_rmse(model: Pipeline, X: pd.DataFrame, y: pd.Series, cv: int) -> np.ndarray:
    """cv分割の交差検証RMSEを配列で返す（scoringの符号を反転させる）。"""

    neg_rmses = cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=cv)
    return -neg_rmses


def summarize_search_results(search: RandomizedSearchCV) -> pd.DataFrame:
    """cv_results_から、探索したパラメータの組み合わせとRMSEの平均・標準偏差を抜き出す。"""

    cv_res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    param_cols = [col for col in cv_res.columns if col.startswith("param_model__")]
    cv_res = cv_res[param_cols + ["mean_test_score", "std_test_score", "rank_test_score"]].copy()
    cv_res = cv_res.rename(columns={col: col.removeprefix("param_model__") for col in param_cols})
    cv_res["mean_test_rmse"] = -cv_res.pop("mean_test_score")
    cv_res = cv_res.rename(columns={"std_test_score": "std_test_rmse", "rank_test_score": "rank"})
    return cv_res.reset_index(drop=True)


def summarize_feature_importances(best_model: Pipeline) -> pd.DataFrame:
    """探索後のrandom_forestの特徴量重要度を、変換後の列名と対応づけて重要度順に返す。"""

    importances = best_model["model"].feature_importances_
    feature_names = [name.removeprefix("num__") for name in best_model["preprocessing"].get_feature_names_out()]
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def compare_importance_methods(
    best_model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    """不純度ベース重要度（MDI）とPermutation Importanceを、同じモデルインスタンスで比較する。

    `best_model`（RandomizedSearchCVが全データで再学習した最終モデル）をそのまま
    Permutation Importanceに使うと、シャッフルして性能低下を測るデータも学習済みで
    リークするため、ここでは`clone()`でハイパーパラメータだけを引き継いだ別インスタンスを
    train/testに分割したtrainで学習し直し、testに対してPermutation Importanceを計算する
    （MDIもこの学習し直したモデルのものを使い、比較の前提を揃える）。
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = clone(best_model)
    model.fit(X_train, y_train)

    feature_names = [name.removeprefix("num__") for name in model["preprocessing"].get_feature_names_out()]
    mdi = model["model"].feature_importances_

    perm_result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, scoring="neg_root_mean_squared_error"
    )

    comparison_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mdi_importance": mdi,
            "permutation_importance": perm_result.importances_mean,
            "permutation_std": perm_result.importances_std,
        }
    )
    comparison_df["mdi_share"] = comparison_df["mdi_importance"] / comparison_df["mdi_importance"].sum()
    positive_sum = comparison_df["permutation_importance"].clip(lower=0).sum()
    comparison_df["permutation_share"] = (
        comparison_df["permutation_importance"].clip(lower=0) / positive_sum if positive_sum > 0 else 0.0
    )
    comparison_df["rank_mdi"] = comparison_df["mdi_importance"].rank(ascending=False, method="min").astype(int)
    comparison_df["rank_permutation"] = (
        comparison_df["permutation_importance"].rank(ascending=False, method="min").astype(int)
    )
    comparison_df["rank_diff"] = comparison_df["rank_mdi"] - comparison_df["rank_permutation"]
    return comparison_df.sort_values("mdi_importance", ascending=False).reset_index(drop=True)


def _plot_model_comparison(
    scores: pd.DataFrame, target_col: str, player_type: str, year: int, cv: int
) -> plt.Figure:
    model_names = [LINEAR_REGRESSION, RANDOM_FOREST]
    groups = [scores.loc[scores["model"] == name, "rmse"] for name in model_names]

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

    ax.set_title(
        f"{year} {player_type} / predicting {target_col}: model comparison ({cv}-fold CV)",
        loc="left",
        fontsize=10,
    )
    ax.set_ylabel("RMSE (out-of-sample)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def _plot_feature_importance(
    importance_df: pd.DataFrame, target_col: str, player_type: str, year: int
) -> plt.Figure:
    ordered = importance_df.sort_values("importance")

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(ordered) + 1)))
    ax.barh(ordered["feature"], ordered["importance"], color=COLOR_PRIMARY, height=0.6)

    ax.set_title(
        f"{year} {player_type} / feature importance for predicting {target_col} (tuned random_forest)",
        loc="left",
        fontsize=10,
    )
    ax.set_xlabel("importance")
    ax.set_axisbelow(True)
    ax.grid(axis="x", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def _plot_importance_comparison(
    comparison_df: pd.DataFrame, target_col: str, player_type: str, year: int
) -> plt.Figure:
    """MDIとPermutation Importanceを、それぞれ合計1に正規化したshareで並べて比較する
    （生の値は目的関数の単位が異なる＝MDIは不純度減少の合計比率、Permutationは
    シャッフル時のRMSE増加量であり、スケールを揃えないと比較しにくいため）。
    """

    ordered = comparison_df.sort_values("mdi_share").reset_index(drop=True)
    y_pos = np.arange(len(ordered))
    height = 0.38

    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(ordered) + 1)))
    ax.barh(y_pos - height / 2, ordered["mdi_share"], height, label="MDI (impurity)", color=COLOR_PRIMARY)
    ax.barh(
        y_pos + height / 2, ordered["permutation_share"], height, label="permutation", color=COLOR_SECONDARY
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered["feature"])
    ax.set_xlabel("relative importance (each method normalized to sum 1 independently)")
    ax.set_title(
        f"{year} {player_type} / MDI vs permutation importance for predicting {target_col}",
        loc="left",
        fontsize=10,
    )
    ax.legend()
    ax.set_axisbelow(True)
    ax.grid(axis="x", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def _plot_partial_dependence(
    best_model: Pipeline, X: pd.DataFrame, features: list[str], target_col: str, player_type: str, year: int
) -> plt.Figure:
    # 整数dtypeの列（p_game等）があるとpartial_dependenceが暗黙の丸め込みを警告してエラーになるため、float化する。
    display = PartialDependenceDisplay.from_estimator(
        best_model, X.astype(float), features=features, kind="average", n_cols=min(3, len(features))
    )
    display.figure_.suptitle(
        f"{year} {player_type} / partial dependence for predicting {target_col} (tuned random_forest)",
        x=0.02,
        ha="left",
        fontsize=10,
    )
    display.figure_.tight_layout()
    return display.figure_


@app.command()
def compare_models(
    year: int = typer.Option(2026, "--year", "-y", help="対象の年"),
    player_type: str = typer.Option("pitcher", "--type", "-t", help="対象（pitcher または batter）"),
    feature_field: list[SavantField] = typer.Option(
        DEFAULT_FEATURES, "--feature", "-f", help="説明変数のフィールド。複数指定可（-f ff_avg_speed -f p_game ...）"
    ),
    target_field: SavantField = typer.Option(SavantField.ERA, "--target-field", help="目的変数のフィールド"),
    cv: int = typer.Option(5, "--cv", help="モデル比較・ハイパーパラメータ探索に使う交差検証の分割数"),
    n_iter: int = typer.Option(20, "--n-iter", help="RandomizedSearchCVの試行回数"),
    random_state: int = typer.Option(42, "--random-state", help="random_forest・探索に使う乱数シード"),
    importance_test_size: float = typer.Option(
        0.2, "--importance-test-size", help="Permutation Importance用に切り出すホールドアウトの割合"
    ),
    n_repeats: int = typer.Option(30, "--n-repeats", help="permutation_importanceのシャッフル回数"),
    pdp_top_n: int = typer.Option(
        4, "--pdp-top-n", help="Partial Dependence Plotを描く特徴量数（permutation importance上位から選ぶ）"
    ),
) -> None:
    """複数の投球フィールドを特徴量にして目的変数（防御率など）を予測する回帰を、
    線形回帰とrandom_forestで比較し、有力なrandom_forestのハイパーパラメータを
    RandomizedSearchCVで探索、特徴量重要度（MDI）を確認する。さらにpermutation_importance
    との比較、Partial Dependence Plotを確認したうえでモデルをjoblibで保存する。

    事前に `data_produce/fetch_leaderboard.py` で `--field`（feature分すべて）と
    `--target-field` を組み合わせて取得しておく必要がある。
    """

    if player_type not in ("pitcher", "batter"):
        raise typer.BadParameter("typeは'pitcher'か'batter'を指定してください")
    if target_field in feature_field:
        raise typer.BadParameter("target-fieldはfeatureに含めないでください")
    if len(set(feature_field)) != len(feature_field):
        raise typer.BadParameter("featureに重複したフィールドが含まれています")
    if len(feature_field) < 2:
        raise typer.BadParameter("featureは2つ以上指定してください（多変量回帰のため）")
    if cv < 2:
        raise typer.BadParameter("cvは2以上を指定してください")
    if n_iter < 1:
        raise typer.BadParameter("n_iterは1以上を指定してください")
    if not 0 < importance_test_size < 1:
        raise typer.BadParameter("importance-test-sizeは0より大きく1未満を指定してください")
    if n_repeats < 1:
        raise typer.BadParameter("n_repeatsは1以上を指定してください")
    if pdp_top_n < 1:
        raise typer.BadParameter("pdp-top-nは1以上を指定してください")

    logger = setup_logger(Path(__file__).stem)
    _apply_style()

    stat_key = savant_client.stat_key_for(list(feature_field) + [target_field])
    csv_name = f"leaderboard/{player_type}/{year}/{stat_key}.csv"
    df = read_csv(csv_name)

    feature_cols = [field.value for field in feature_field]
    target_col = target_field.value
    data = df[feature_cols + [target_col]].copy()
    # 球種使用率（n_*_formatted）のNaNは「その球種を投げていない」＝0%であり欠損ではないため、
    # dropnaで行ごと落とす前に0埋めする（そうしないと全球種を投げる投手だけに標本が偏る）。
    usage_cols = [col for col in feature_cols if col.startswith("n_") and col.endswith("_formatted")]
    data[usage_cols] = data[usage_cols].fillna(0)
    data = data.dropna()

    if len(data) <= cv:
        raise typer.BadParameter(f"データ数（n={len(data)}）がcv（{cv}）以下です。cvを小さくするかデータを増やしてください")

    X = data[feature_cols]
    y = data[target_col]

    logger.info(f"=== [{stat_key}] {csv_name} (n={len(data)}) ===")
    logger.info(f"特徴量({len(feature_cols)}件): {feature_cols}")
    logger.info(f"目的変数: {target_col}")

    logger.info(f"\n=== モデル比較（{cv}-fold cross_val_score, RMSE） ===")
    models = build_models(feature_cols, random_state)
    comparison_rows = []
    for name, model in models.items():
        rmses = cross_val_rmse(model, X, y, cv)
        comparison_rows += [{"model": name, "fold": fold, "rmse": rmse} for fold, rmse in enumerate(rmses)]
        logger.info(f"{name}: mean={rmses.mean():.4f}  std={rmses.std():.4f}  min={rmses.min():.4f}  max={rmses.max():.4f}")
    comparison_scores = pd.DataFrame(comparison_rows)

    means = comparison_scores.groupby("model")["rmse"].mean()
    winner = means.idxmin()
    logger.info(f"cross_val_score平均RMSEが最小のモデル: {winner}（{means[winner]:.4f}）")

    comparison_fig = _plot_model_comparison(comparison_scores, target_col, player_type, year, cv)
    comparison_fig_path = save_figure(comparison_fig, f"{player_type}/{year}/{stat_key}_compare_models.png", dpi=150)
    plt.close(comparison_fig)
    comparison_table_path = save_table(comparison_scores, f"{player_type}/{year}/{stat_key}_compare_models.csv")
    logger.info(f"図を保存しました: {comparison_fig_path}")
    logger.info(f"表を保存しました: {comparison_table_path}")

    logger.info(f"\n=== RandomizedSearchCV（random_forestのハイパーパラメータ探索, n_iter={n_iter}, cv={cv}） ===")
    logger.info(
        "cross_val比較の結果によらずrandom_forestを深掘りする"
        "（決定木の集合であるrandom_forestにはn_estimators等の探索余地があるが、線形回帰にはほぼない）。"
    )
    param_distribs = {
        "model__n_estimators": randint(50, 400),
        "model__max_depth": randint(2, 20),
        "model__max_features": uniform(0.1, 0.9),
        "model__min_samples_leaf": randint(1, 10),
    }
    rnd_search = RandomizedSearchCV(
        models[RANDOM_FOREST],
        param_distributions=param_distribs,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
    )
    rnd_search.fit(X, y)
    best_model = rnd_search.best_estimator_
    logger.info(f"最良パラメータ: {rnd_search.best_params_}")
    logger.info(f"最良RMSE: {-rnd_search.best_score_:.4f}")

    search_results = summarize_search_results(rnd_search)
    logger.info(f"探索結果（上位5件）:\n{search_results.head(5).to_string(index=False)}")
    search_table_path = save_table(search_results, f"{player_type}/{year}/{stat_key}_tune_random_forest.csv")
    logger.info(f"表を保存しました: {search_table_path}")

    logger.info("\n=== 特徴量重要度（探索後のrandom_forest） ===")
    importance_df = summarize_feature_importances(best_model)
    logger.info(f"重要度順:\n{importance_df.to_string(index=False)}")
    importance_fig = _plot_feature_importance(importance_df, target_col, player_type, year)
    importance_fig_path = save_figure(importance_fig, f"{player_type}/{year}/{stat_key}_feature_importance.png", dpi=150)
    plt.close(importance_fig)
    importance_table_path = save_table(importance_df, f"{player_type}/{year}/{stat_key}_feature_importance.csv")
    logger.info(f"図を保存しました: {importance_fig_path}")
    logger.info(f"表を保存しました: {importance_table_path}")

    logger.info(
        f"\n=== MDI vs Permutation Importance（test_size={importance_test_size}, n_repeats={n_repeats}） ==="
    )
    logger.info(
        "不純度ベース重要度（MDI、分岐に使われた頻度）に対し、"
        "permutation_importance（ホールドアウトで列をシャッフルした際のRMSE悪化）を比較する。"
        "best_modelは全データで再学習済みのためそのまま使うとリークするので、"
        "ハイパーパラメータのみ引き継いだ別インスタンスをtrain/testに分割して学習し直す。"
    )
    importance_comparison_df = compare_importance_methods(
        best_model, X, y, importance_test_size, n_repeats, random_state
    )
    logger.info(f"重要度比較（MDI順）:\n{importance_comparison_df.to_string(index=False)}")
    n_moved = (importance_comparison_df["rank_diff"].abs() >= 2).sum()
    logger.info(
        f"順位が2以上動いた特徴量は{n_moved}件（全{len(importance_comparison_df)}件中）。"
        "MDIは訓練時の分岐頻度、Permutationはテスト性能への寄与という異なる基準のため、"
        "一致しない場合は片方だけを鵜呑みにしないことが重要。"
    )
    comparison_importance_fig = _plot_importance_comparison(importance_comparison_df, target_col, player_type, year)
    comparison_importance_fig_path = save_figure(
        comparison_importance_fig, f"{player_type}/{year}/{stat_key}_importance_comparison.png", dpi=150
    )
    plt.close(comparison_importance_fig)
    comparison_importance_table_path = save_table(
        importance_comparison_df, f"{player_type}/{year}/{stat_key}_importance_comparison.csv"
    )
    logger.info(f"図を保存しました: {comparison_importance_fig_path}")
    logger.info(f"表を保存しました: {comparison_importance_table_path}")

    logger.info(f"\n=== Partial Dependence Plot（permutation importance上位{pdp_top_n}件） ===")
    pdp_features = (
        importance_comparison_df.sort_values("rank_permutation").head(pdp_top_n)["feature"].tolist()
    )
    logger.info(f"対象特徴量: {pdp_features}")
    pdp_fig = _plot_partial_dependence(best_model, X, pdp_features, target_col, player_type, year)
    pdp_fig_path = save_figure(pdp_fig, f"{player_type}/{year}/{stat_key}_partial_dependence.png", dpi=150)
    plt.close(pdp_fig)
    logger.info(f"図を保存しました: {pdp_fig_path}")

    logger.info("\n=== モデルの保存と再読込（joblib） ===")
    model_path = output_path(f"{player_type}/{year}/{stat_key}_best_random_forest.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    record_write(model_path)
    reloaded_model = joblib.load(model_path)
    record_read(model_path)
    original_pred = best_model.predict(X.head(5))
    reloaded_pred = reloaded_model.predict(X.head(5))
    match = np.allclose(original_pred, reloaded_pred)
    logger.info(f"保存先: {model_path}")
    logger.info(f"保存前と再読込後の予測が一致: {match}")


if __name__ == "__main__":
    app()
