"""不純度ベースの特徴量重要度（feature_importances_）とPermutation Importance
（sklearn.inspection.permutation_importance）を、同じ最終モデルに対して計算し、
順位・値を並べて比較する実験。

背景: 10_finalize_model.pyでは`feature_importances_`（分岐に使われた頻度に基づく
不純度ベースの重要度、MDI）だけを見ており、README.mdの「未学習の概念」に
「特徴量重要度の裏付け（Permutation ImportanceやPartial Dependenceとの比較は
未実施）」として挙がっていた。ここでは10で保存したfinal_model.pklをそのまま
読み込み、テストセットに対してPermutation Importanceを追加計算する。
比較の粒度をMDIと揃えるため、前処理後の変換済み特徴量（`bedrooms__ratio`や
`cat__ocean_proximity_INLAND`など）に対してPermutation Importanceを計算する
（前処理込みPipeline全体に対して生の列を並べ替えるのではない）。

【実行前に予想】不純度ベースの重要度は高カーディナリティなOne-Hot後の
ocean_proximity各列（cat__ocean_proximity_INLANDなど）を過大評価しやすいと
言われる。実際に順位が入れ替わる特徴量はあるか？
また、KMeansクラスタ類似度特徴量（n_clusters=45、いずれも緯度・経度から
作られており互いに強く相関するはず）は、不純度ベースでは合計51.7%
（10_finalize_model.pyのログより）を占めていた。Permutation Importanceで
1列ずつ独立に並べ替えると、残り44列が同じような情報を代わりに提供できて
しまうため、個々のクラスタ列の重要度は不純度ベースより過小評価される
のではないか（＝MDIの「高カーディナリティ過大評価」とは逆に、Permutation
Importance側は「強く相関する特徴量群」を過小評価するという、別方向の
バイアスが見えるのではないか）。
"""

from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "learn"))
from common import HOUSING_TEST_CSV, load_features_and_labels  # noqa: E402
from data_produce.util.logging_config import setup_logger  # noqa: E402
from data_produce.util.metadata import record_read  # noqa: E402

MODEL_PATH = PROJECT_DIR / "learn" / "outputs" / "final_model.pkl"
N_REPEATS = 10
RANDOM_STATE = 42
TOP_N = 20


def load_final_model() -> Pipeline:
    """10_finalize_model.pyが保存した最終モデルをそのまま読み込む。"""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"{MODEL_PATH}が見つかりません。先にlearn/10_finalize_model.pyを実行してください。"
        )
    model = joblib.load(MODEL_PATH)
    record_read(MODEL_PATH)
    return model


def _feature_group(feature_name: str) -> str:
    """変換後の列名から、元の列・対数列・比率列・クラスタ類似度列・カテゴリ列を判定する
    （10_finalize_model.pyの_feature_groupと同じ分類基準）。
    """

    prefix = feature_name.split("__", 1)[0]
    if prefix in ("bedrooms", "rooms_per_house", "people_per_house"):
        return "比率列"
    if prefix == "log":
        return "対数列"
    if prefix == "geo":
        return "クラスタ類似度列"
    if prefix == "cat":
        return "カテゴリ列（One-Hot）"
    return "元の列（remainder）"


def compute_importances(
    final_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """変換後の特徴量空間を揃えた上で、不純度ベースとPermutation Importanceを計算する。"""

    preprocessing = final_model.named_steps["preprocessing"]
    regressor = final_model.named_steps["random_forest"]

    X_test_transformed = preprocessing.transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
    feature_names = preprocessing.get_feature_names_out()

    impurity = regressor.feature_importances_

    perm_result = permutation_importance(
        regressor, X_test_transformed, y_test,
        n_repeats=N_REPEATS, random_state=RANDOM_STATE, n_jobs=1, scoring="r2",
    )

    df = pd.DataFrame({
        "feature": feature_names,
        "group": [_feature_group(name) for name in feature_names],
        "impurity": impurity,
        "permutation": perm_result.importances_mean,
        "permutation_std": perm_result.importances_std,
    })
    df["impurity_share"] = df["impurity"] / df["impurity"].sum()
    positive_sum = df["permutation"].clip(lower=0).sum()
    df["permutation_share"] = df["permutation"].clip(lower=0) / positive_sum
    df["rank_impurity"] = df["impurity"].rank(ascending=False, method="min").astype(int)
    df["rank_permutation"] = df["permutation"].rank(ascending=False, method="min").astype(int)
    df["rank_diff"] = df["rank_impurity"] - df["rank_permutation"]
    return df.sort_values("impurity", ascending=False).reset_index(drop=True)


GROUP_LABELS_EN = {
    "比率列": "ratio cols",
    "対数列": "log cols",
    "クラスタ類似度列": "cluster similarity cols",
    "カテゴリ列（One-Hot）": "one-hot cat cols",
    "元の列（remainder）": "remainder cols",
}


def plot_comparison(df: pd.DataFrame, outputs_dir: Path) -> Path:
    """上位特徴量の並びとグループ合計を、不純度ベース・Permutationで並べて描く。
    matplotlibのデフォルトフォントは日本語グリフを持たないため、図中のラベルは
    英語表記に変換する（ログ・docstringの日本語表記はGROUP_LABELS_ENのキー側）。
    """

    top = df.head(TOP_N).iloc[::-1]
    y = np.arange(len(top))
    height = 0.38

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 11))

    ax1.barh(y - height / 2, top["impurity_share"], height, label="impurity (MDI)")
    ax1.barh(y + height / 2, top["permutation_share"], height, label="permutation")
    ax1.set_yticks(y)
    ax1.set_yticklabels(top["feature"])
    ax1.set_xlabel("relative importance (each method normalized to sum 1 independently)")
    ax1.set_title(f"feature-level comparison (top {TOP_N} by impurity)")
    ax1.legend()
    ax1.grid(True, axis="x")

    group_summary = (
        df.groupby("group")[["impurity_share", "permutation_share"]]
        .sum()
        .sort_values("impurity_share", ascending=False)
    )
    x = np.arange(len(group_summary))
    width = 0.35
    ax2.bar(x - width / 2, group_summary["impurity_share"], width, label="impurity (MDI)")
    ax2.bar(x + width / 2, group_summary["permutation_share"], width, label="permutation")
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [GROUP_LABELS_EN[name] for name in group_summary.index], rotation=20, ha="right"
    )
    ax2.set_ylabel("relative importance (group sum)")
    ax2.set_title("group-level comparison")
    ax2.legend()
    ax2.grid(True, axis="y")

    fig.tight_layout()
    path = outputs_dir / "importance_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    setup_logger(Path(__file__).stem)
    outputs_dir = Path(__file__).resolve().parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    final_model = load_final_model()
    X_test, y_test = load_features_and_labels(HOUSING_TEST_CSV)
    logger.info(f"テストセット: {X_test.shape[0]:,}行 x {X_test.shape[1]}列")
    logger.info(
        f"Permutation Importance: n_repeats={N_REPEATS}, scoring=r2, "
        f"random_state={RANDOM_STATE}"
    )

    df = compute_importances(final_model, X_test, y_test)

    logger.info("\n=== 上位10件（不純度ベース順） ===")
    logger.info(
        "\n"
        + df.head(10)[
            ["feature", "impurity_share", "permutation_share", "rank_impurity", "rank_permutation", "rank_diff"]
        ].to_string(index=False)
    )

    logger.info("\n=== グループ別の重要度合計 ===")
    group_summary = df.groupby("group")[["impurity_share", "permutation_share"]].sum()
    logger.info(f"\n{group_summary.sort_values('impurity_share', ascending=False).to_string()}")

    logger.info("\n=== ocean_proximity（One-Hot列）の順位変化 ===")
    cat_rows = df[df["group"] == "カテゴリ列（One-Hot）"]
    logger.info(f"\n{cat_rows[['feature', 'impurity_share', 'permutation_share', 'rank_impurity', 'rank_permutation', 'rank_diff']].to_string(index=False)}")

    logger.info("\n=== 順位が大きく動いた特徴量（|rank_diff|の上位10件） ===")
    biggest_movers = df.reindex(df["rank_diff"].abs().sort_values(ascending=False).index).head(10)
    logger.info(
        f"\n{biggest_movers[['feature', 'group', 'rank_impurity', 'rank_permutation', 'rank_diff']].to_string(index=False)}"
    )

    path = plot_comparison(df, outputs_dir)
    logger.info(f"\n保存しました: {path}")

    logger.info("\n=== まとめ ===")
    cat_impurity_share = group_summary.loc["カテゴリ列（One-Hot）", "impurity_share"]
    cat_permutation_share = group_summary.loc["カテゴリ列（One-Hot）", "permutation_share"]
    logger.info(
        f"カテゴリ列（One-Hot）の重要度合計は impurity={cat_impurity_share:.1%} → "
        f"permutation={cat_permutation_share:.1%}"
        f"（{'permutationの方が高い' if cat_permutation_share > cat_impurity_share else 'permutationの方が低い、または同水準'}）。"
    )
    geo_impurity_share = group_summary.loc["クラスタ類似度列", "impurity_share"]
    geo_permutation_share = group_summary.loc["クラスタ類似度列", "permutation_share"]
    logger.info(
        f"クラスタ類似度列（互いに強く相関する45列）の重要度合計は "
        f"impurity={geo_impurity_share:.1%} → permutation={geo_permutation_share:.1%}"
        f"（{'予想通りpermutationの方が過小評価' if geo_permutation_share < geo_impurity_share else '予想に反してpermutationの方が高い、または同水準'}）。"
        "1列ずつ独立に並べ替えるPermutation Importanceは、相関する特徴量群のうち"
        "並べ替えた1列の情報が他の列で代替されてしまう分、個々の列・グループの重要度を"
        "過小評価しやすいという性質を持つ。"
    )
    n_moved = (df["rank_diff"].abs() >= 5).sum()
    logger.info(
        f"順位が5以上動いた特徴量は{n_moved}件（全{len(df)}件中）。"
        "不純度ベースとPermutation Importanceは異なる前提（訓練データでの分岐頻度 vs "
        "テストデータでの予測性能への寄与）で重要度を測っているため、順位はある程度"
        "入れ替わるのが自然であり、片方だけを鵜呑みにしないことが重要。"
    )


if __name__ == "__main__":
    main()
