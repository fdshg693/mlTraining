"""09_tune_model.pyのRandomized Searchで選ばれた最良モデルを解釈・最終評価・保存する。

09と同じrun_randomized_search（random_state=42固定）を再実行してbest_estimator_を
得ることで、探索結果を再現しつつ、このファイルでは特徴量重要度の解釈・
テストセットでの一度きりの評価・モデル永続化に集中する。
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_DIR))
from common import HOUSING_TEST_CSV, build_full_preprocessing, load_features_and_labels
from data_produce.util.file_io import output_path
from data_produce.util.logging_config import setup_logger
from data_produce.util.metadata import record_read, record_write

CV_FOLDS = 3
N_ITER = 10
RANDOM_STATE = 42
CONFIDENCE = 0.95
MODEL_FILE_NAME = "final_model.pkl"


def search_best_model(housing: pd.DataFrame, housing_labels: pd.Series) -> Pipeline:
    """09_tune_model.pyと同じRandomized Searchを再実行し、best_estimator_を返す。

    random_state=42で固定しているため、09で確認した
    n_clusters=45, max_features=9が同じ順序で再現される。
    """

    full_pipeline = Pipeline([
        ("preprocessing", build_full_preprocessing()),
        ("random_forest", RandomForestRegressor(random_state=RANDOM_STATE)),
    ])
    param_distribs = {
        "preprocessing__geo__n_clusters": randint(low=3, high=50),
        "random_forest__max_features": randint(low=2, high=20),
    }
    rnd_search = RandomizedSearchCV(
        full_pipeline, param_distributions=param_distribs, n_iter=N_ITER,
        cv=CV_FOLDS, scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
    )
    rnd_search.fit(housing, housing_labels)
    return rnd_search.best_estimator_


def _feature_group(feature_name: str) -> str:
    """変換後の列名から、元の列・対数列・比率列・クラスタ類似度列・カテゴリ列を判定する。"""

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


def summarize_feature_importances(final_model: Pipeline) -> pd.DataFrame:
    """特徴量重要度を列名・グループと対応づけ、重要度の高い順に並べる。"""

    importances = final_model["random_forest"].feature_importances_
    feature_names = final_model["preprocessing"].get_feature_names_out()
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
        "group": [_feature_group(name) for name in feature_names],
    })
    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def evaluate_on_test_set(
    final_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[np.ndarray, float]:
    """それまで触らなかったテストセットに一度だけ予測し、RMSEを計算する。"""

    predictions = final_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    return predictions, rmse


def bootstrap_rmse_interval(
    predictions: np.ndarray, y_test: pd.Series, confidence: float = CONFIDENCE
) -> tuple[float, float]:
    """テストRMSEの二乗誤差をブートストラップし、信頼区間を計算する。"""

    def rmse_stat(squared_errors: np.ndarray) -> float:
        return np.sqrt(np.mean(squared_errors))

    squared_errors = (predictions - y_test.to_numpy()) ** 2
    boot_result = stats.bootstrap(
        [squared_errors], rmse_stat, confidence_level=confidence, random_state=RANDOM_STATE
    )
    return boot_result.confidence_interval.low, boot_result.confidence_interval.high


def save_and_reload_model(final_model: Pipeline) -> Path:
    """モデルをjoblibで保存し、再読込後の予測が保存前と一致することを確認する。"""

    path = output_path(MODEL_FILE_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, path)
    record_write(path)

    reloaded_model = joblib.load(path)
    record_read(path)
    return reloaded_model, path


def main() -> None:
    setup_logger(Path(__file__).stem)
    housing, housing_labels = load_features_and_labels()
    logger.info(f"訓練セット（特徴量）: {housing.shape[0]:,}行 x {housing.shape[1]}列")

    logger.info(f"\n=== Randomized Search再実行（n_iter={N_ITER}, cv={CV_FOLDS}） ===")
    final_model = search_best_model(housing, housing_labels)
    best_n_clusters = final_model.named_steps["preprocessing"].named_transformers_["geo"].n_clusters
    best_max_features = final_model.named_steps["random_forest"].max_features
    logger.info(f"最良パラメータ: n_clusters={best_n_clusters}, max_features={best_max_features}")

    logger.info("\n=== 特徴量重要度 ===")
    importance_df = summarize_feature_importances(final_model)
    logger.info(f"上位10件:\n{importance_df.head(10).to_string(index=False)}")
    logger.info(f"グループ別の重要度合計:\n{importance_df.groupby('group')['importance'].sum().sort_values(ascending=False)}")
    logger.info(
        "重要度はランダムフォレストがどの特徴量を分岐に多く使ったかの手がかりであり、"
        "その特徴量が価格を「引き上げる/下げる」といった因果効果を示すものではない。"
    )

    logger.info("\n=== テストセットでの最終評価（一度だけ） ===")
    X_test, y_test = load_features_and_labels(HOUSING_TEST_CSV)
    logger.info(f"テストセット: {X_test.shape[0]:,}行 x {X_test.shape[1]}列")
    predictions, test_rmse = evaluate_on_test_set(final_model, X_test, y_test)
    logger.info(f"テストRMSE: {test_rmse:,.0f}")

    rmse_lower, rmse_upper = bootstrap_rmse_interval(predictions, y_test)
    logger.info(
        f"{int(CONFIDENCE * 100)}%信頼区間: [{rmse_lower:,.0f}, {rmse_upper:,.0f}]"
    )
    logger.info(
        "テストRMSE単体だけでなく、標本の取り方によるばらつきを信頼区間として"
        "併記することで、単一の数値の見かけ上の精度に過信しないようにする。"
    )

    logger.info("\n=== モデルの保存と再読込 ===")
    reloaded_model, model_path = save_and_reload_model(final_model)
    logger.info(f"保存しました: {model_path}")
    original_predictions = final_model.predict(X_test.iloc[:5])
    reloaded_predictions = reloaded_model.predict(X_test.iloc[:5])
    match = np.allclose(original_predictions, reloaded_predictions)
    logger.info(f"保存前と再読込後の予測が一致: {match}")
    logger.info(f"保存前の予測（先頭5件）: {original_predictions.round(-2)}")
    logger.info(f"再読込後の予測（先頭5件）: {reloaded_predictions.round(-2)}")

    logger.info("\n=== 推論に必要な入力形式 ===")
    logger.info(f"列名: {X_test.columns.tolist()}")
    logger.info(f"dtype:\n{X_test.dtypes}")
    logger.info(
        "median_house_value（目的変数）を除いた上記の列・型で新しい地区データを渡せば、"
        "保存済みのfinal_model（前処理込み）がそのまま予測を返す。"
        "列の欠損やocean_proximityの未知カテゴリはPipeline内のSimpleImputer/"
        "OneHotEncoder(handle_unknown='ignore')が吸収する。"
    )


if __name__ == "__main__":
    main()
