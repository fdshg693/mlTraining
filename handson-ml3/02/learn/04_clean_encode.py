"""訓練セットの欠損値・外れ値・カテゴリ値を処理する。

target(median_house_value)は前処理の対象から外し、特徴量側だけを扱う。
統計量（中央値など）は訓練データで`fit`し、同じ統計量を使い回す前提を守る。
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

PROJECT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_DIR))
from data_produce.util.file_io import data_path, read_csv
from data_produce.util.logging_config import setup_logger

HOUSING_TRAIN_CSV = "housing_train.csv"
TARGET_COLUMN = "median_house_value"
CATEGORICAL_COLUMN = "ocean_proximity"


def load_features_and_labels() -> tuple[pd.DataFrame, pd.Series]:
    """02_split_data.pyが保存した訓練セットを読み込み、特徴量とラベルに分ける。"""

    if not data_path(HOUSING_TRAIN_CSV).is_file():
        raise FileNotFoundError(
            f"{HOUSING_TRAIN_CSV}が見つかりません。"
            "先にlearn/02_split_data.pyを実行してください。"
        )
    housing = read_csv(HOUSING_TRAIN_CSV)
    housing_labels = housing[TARGET_COLUMN].copy()
    housing_features = housing.drop(columns=[TARGET_COLUMN])
    return housing_features, housing_labels


def compare_missing_value_options(
    housing: pd.DataFrame,
    column: str = "total_bedrooms",
) -> dict[str, pd.DataFrame]:
    """欠損値への3つの対処（行削除・列削除・中央値補完）を別々のコピーで試す。"""

    null_rows_idx = housing[column].isnull()

    option1_drop_rows = housing.copy()
    option1_drop_rows = option1_drop_rows.dropna(subset=[column])

    option2_drop_column = housing.copy()
    option2_drop_column = option2_drop_column.drop(columns=[column])

    option3_fill_median = housing.copy()
    median = housing[column].median()
    option3_fill_median[column] = option3_fill_median[column].fillna(median)

    return {
        "行削除": option1_drop_rows,
        "列削除": option2_drop_column,
        "中央値補完": option3_fill_median,
        "欠損だった行数": null_rows_idx.sum(),
    }


def impute_numeric_median(housing: pd.DataFrame) -> tuple[pd.DataFrame, SimpleImputer]:
    """数値列に対してSimpleImputer(strategy="median")をfit_transformする。"""

    housing_num = housing.select_dtypes(include=[np.number])
    imputer = SimpleImputer(strategy="median")
    array = imputer.fit_transform(housing_num)
    housing_tr = pd.DataFrame(array, columns=housing_num.columns, index=housing_num.index)
    return housing_tr, imputer


def detect_outliers(housing_num_imputed: pd.DataFrame) -> np.ndarray:
    """IsolationForestで外れ値候補を検出する（削除はしない）。

    戻り値は1（正常）/-1（外れ値候補）のラベル配列。
    ここでは検出結果を確認するだけにとどめ、削除がモデル性能や標本分布に
    与える影響は別途の実験で検証する対象とする。
    """

    # IsolationForest: ランダムな特徴量・分割値で決定木を多数生成し、
    # 各サンプルが葉に到達するまでの分割回数（パス長）で異常度を測る手法。
    # 外れ値は他と値が離れているため少ない分割で孤立しやすく、パス長が短くなる
    # （正常データは多数派に埋もれて分離に多くの分割を要する）。
    # fit_predictは1（正常）/-1（外れ値）を返す。
    isolation_forest = IsolationForest(random_state=42)
    return isolation_forest.fit_predict(housing_num_imputed)


def encode_ordinal(housing_cat: pd.DataFrame) -> tuple[np.ndarray, OrdinalEncoder]:
    """OrdinalEncoderでカテゴリ列を整数化する（名義尺度に順序を与える危険の確認用）。"""

    # OrdinalEncoder: 各カテゴリ文字列を出現順（fit時にソートされた順）に
    # 0, 1, 2, ... の整数へ置き換えるscikit-learnの前処理クラス。
    # ここでの役割はocean_proximity（本質的には順序のないカテゴリ）を
    # あえてこのエンコーダに通し、モデルからは「1と2は近く、1と3は遠い」
    # といった実在しない順序・距離関係があるかのように見えてしまう問題を
    # 後段のログ出力（173-175行目）で確認するためのもの。
    # 実運用ではこの後のOneHotEncoderのように順序を仮定しない変換を使う。
    ordinal_encoder = OrdinalEncoder()
    encoded = ordinal_encoder.fit_transform(housing_cat)
    return encoded, ordinal_encoder


def encode_one_hot(housing_cat: pd.DataFrame) -> tuple[pd.DataFrame, OneHotEncoder]:
    """OneHotEncoder(handle_unknown="ignore")でカテゴリ列を疎行列からDataFrameへ変換する。"""

    # OneHotEncoder: 各カテゴリごとに0/1の列を1つずつ用意し、該当する列だけ1、
    # それ以外を0にする（例: ocean_proximityの5カテゴリなら5列に展開）。
    # OrdinalEncoderと違い整数の大小関係を持ち込まないため、名義尺度である
    # ocean_proximityに対してモデルへ実在しない順序・距離を与えずに済む、
    # ここでの「本来使うべき」変換として対比される役割を持つ。
    # handle_unknown="ignore"は、fit時に無かったカテゴリがtransform時に来ても
    # ValueErrorにせず全列0として扱う設定（未知カテゴリの検証は
    # transform_unknown_categoryで行う）。
    cat_encoder = OneHotEncoder(handle_unknown="ignore")
    sparse_matrix = cat_encoder.fit_transform(housing_cat)
    encoded_df = pd.DataFrame(
        sparse_matrix.toarray(),
        columns=cat_encoder.get_feature_names_out(),
        index=housing_cat.index,
    )
    return encoded_df, cat_encoder


def transform_unknown_category(
    cat_encoder: OneHotEncoder,
    unknown_rows: pd.DataFrame,
) -> pd.DataFrame:
    """fit時に無かったカテゴリ値をhandle_unknown="ignore"でどう扱うか確認する。"""

    array = cat_encoder.transform(unknown_rows)
    return pd.DataFrame(
        array.toarray() if hasattr(array, "toarray") else array,
        columns=cat_encoder.get_feature_names_out(),
        index=unknown_rows.index,
    )


def main() -> None:
    setup_logger(Path(__file__).stem)
    housing, _housing_labels = load_features_and_labels()
    logger.info(f"訓練セット（特徴量のみ）: {housing.shape[0]:,}行 x {housing.shape[1]}列")

    logger.info("\n=== 欠損値処理の3案（total_bedrooms） ===")
    options = compare_missing_value_options(housing)
    logger.info(f"欠損だった行数: {options['欠損だった行数']:,}")
    for name in ("行削除", "列削除", "中央値補完"):
        df = options[name]
        logger.info(f"{name}: {df.shape[0]:,}行 x {df.shape[1]}列")
    logger.info(
        "行削除は行数が減り、列削除は情報そのものを失う。"
        "中央値補完は行数・列数を保ったまま欠損だけを埋められるため、"
        "以降はSimpleImputerによる中央値補完を数値列全体へ適用する。"
    )

    logger.info("\n=== SimpleImputer(strategy=\"median\")による数値列の補完 ===")
    housing_tr, imputer = impute_numeric_median(housing)
    logger.info(f"補完後の欠損数（列ごと）:\n{housing_tr.isnull().sum().to_string()}")
    logger.info(f"fit()で学習した中央値（imputer.statistics_）:\n{imputer.statistics_}")
    manual_median = housing.select_dtypes(include=[np.number]).median().to_numpy()
    logger.info(f"手計算の中央値と一致するか: {np.allclose(imputer.statistics_, manual_median)}")

    logger.info("\n=== IsolationForestによる外れ値候補の検出 ===")
    outlier_pred = detect_outliers(housing_tr)
    n_outliers = int((outlier_pred == -1).sum())
    logger.info(f"外れ値候補: {n_outliers:,}行 / {len(outlier_pred):,}行")
    logger.info(
        "ここでは検出結果の確認にとどめ、自動削除はしない。"
        "削除の是非はデータ生成過程の異常か、重要な少数例かを踏まえて別途判断する。"
    )

    housing_cat = housing[[CATEGORICAL_COLUMN]]
    logger.info(f"\n=== {CATEGORICAL_COLUMN}のカテゴリ確認 ===")
    logger.info(f"\n{housing_cat[CATEGORICAL_COLUMN].value_counts().to_string()}")

    logger.info("\n=== OrdinalEncoderによる数値化 ===")
    ordinal_encoded, ordinal_encoder = encode_ordinal(housing_cat)
    logger.info(f"変換後の先頭5件: {ordinal_encoded[:5].ravel()}")
    logger.info(f"categories_: {ordinal_encoder.categories_}")
    logger.info(
        "OrdinalEncoderは0,1,2...の整数を割り当てるだけで、"
        "ocean_proximityのカテゴリ間に順序関係を意図していないにもかかわらず、"
        "モデルからは順序・距離があるように見えてしまう危険がある。"
    )

    logger.info("\n=== OneHotEncoderによる変換 ===")
    one_hot_df, cat_encoder = encode_one_hot(housing_cat)
    logger.info(f"変換後shape: {one_hot_df.shape}")
    logger.info(f"feature_names_in_: {cat_encoder.feature_names_in_}")
    logger.info(f"get_feature_names_out(): {cat_encoder.get_feature_names_out()}")

    logger.info("\n=== 未知カテゴリの扱い（handle_unknown=\"ignore\"） ===")
    known_categories = list(ordinal_encoder.categories_[0])
    unknown_value = "UNKNOWN_CATEGORY"
    unknown_rows = pd.DataFrame({CATEGORICAL_COLUMN: [known_categories[0], unknown_value]})
    transformed = transform_unknown_category(cat_encoder, unknown_rows)
    logger.info(f"\n{transformed.to_string()}")
    logger.info(
        "未知カテゴリの行はすべての列が0になり、変換自体はエラーで止まらない。"
        "handle_unknown=\"ignore\"を指定しない場合はfit時に無い値でValueErrorになるため、"
        "本番推論での未知カテゴリ発生を見越した設定が必要になる。"
    )


if __name__ == "__main__":
    main()
