"""PipelineとColumnTransformerで数値列・カテゴリ列の前処理を一体化する。

06_custom_transformers.pyで作った関数・カスタム変換器を、
「学習時と推論時で必ず同じ変換が適用される」1つのグラフへ組み立て直す。
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from loguru import logger
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import (
    ColumnTransformer,
    make_column_selector,
    make_column_transformer,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

PROJECT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_DIR))
from data_produce.util.file_io import data_path, output_path, read_csv
from data_produce.util.logging_config import setup_logger

HOUSING_TRAIN_CSV = "housing_train.csv"
TARGET_COLUMN = "median_house_value"
NUM_ATTRIBS = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
]
CAT_ATTRIBS = ["ocean_proximity"]


def load_features() -> pd.DataFrame:
    """02_split_data.pyが保存した訓練セットを読み込み、目的変数を除いた特徴量を返す。"""

    if not data_path(HOUSING_TRAIN_CSV).is_file():
        raise FileNotFoundError(
            f"{HOUSING_TRAIN_CSV}が見つかりません。"
            "先にlearn/02_split_data.pyを実行してください。"
        )
    housing = read_csv(HOUSING_TRAIN_CSV)
    return housing.drop(columns=[TARGET_COLUMN])


def build_num_pipeline() -> Pipeline:
    """数値列へ「中央値補完→標準化」を適用するパイプライン。"""

    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


def inspect_pipeline(num_pipeline: Pipeline) -> dict[str, object]:
    """Pipelineをステップ名・インデックス・スライスで操作した結果を集める。"""

    return {
        "steps": [name for name, _ in num_pipeline.steps],
        "second_step": num_pipeline[1],
        "all_but_last": [name for name, _ in num_pipeline[:-1].steps],
        "named_step": num_pipeline.named_steps["simpleimputer"],
    }


def build_cat_pipeline() -> Pipeline:
    """カテゴリ列へ「最頻値補完→未知値を無視するOne-Hot」を適用するパイプライン。"""

    return make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        # handle_unknown="ignore"は欠損値(NaN)ではなく、学習時に存在しなかった未知カテゴリが
        # 推論時に出現した場合の対策。SimpleImputerの補完とは独立した問題なので必要。
        OneHotEncoder(handle_unknown="ignore"),
    )


def build_column_transformer_by_name(
    num_pipeline: Pipeline, cat_pipeline: Pipeline
) -> ColumnTransformer:
    """列名を明示的に指定するColumnTransformer。"""

    return ColumnTransformer([
        ("num", num_pipeline, NUM_ATTRIBS),
        ("cat", cat_pipeline, CAT_ATTRIBS),
    ])


def build_column_transformer_by_dtype(
    num_pipeline: Pipeline, cat_pipeline: Pipeline
) -> ColumnTransformer:
    """dtypeで列を選ぶmake_column_selectorを使ったColumnTransformer。"""

    return make_column_transformer(
        (num_pipeline, make_column_selector(dtype_include=np.number)),
        (cat_pipeline, make_column_selector(dtype_include=object)),
    )


def column_ratio(X: np.ndarray) -> np.ndarray:
    """2列目に対する1列目の比率を1列のarrayとして返す。"""

    return X[:, [0]] / X[:, [1]]


def ratio_name(function_transformer: FunctionTransformer, feature_names_in) -> list[str]:
    return ["ratio"]


def ratio_pipeline() -> Pipeline:
    """欠損補完→列比率→標準化の3ステップパイプライン。"""

    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler(),
    )


def build_log_pipeline() -> Pipeline:
    """欠損補完→対数変換→標準化の3ステップパイプライン。"""

    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler(),
    )


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """KMeansのクラスタ中心とのRBF類似度を特徴量として出力する変換器。

    06_custom_transformers.pyと同じ実装。ファイル名が数字始まりで
    import できないため、各テーマのファイルへそのまま持たせている。
    """

    def __init__(self, n_clusters: int = 10, gamma: float = 1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            self.n_clusters, n_init=10, random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        check_is_fitted(self)
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None) -> list[str]:
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


def build_full_preprocessing(cat_pipeline: Pipeline) -> ColumnTransformer:
    """比率・対数・クラスタ類似度・カテゴリ変換を1つのColumnTransformerへ統合する。

    remainderへ渡した数値パイプラインが、他のどの変換にも指定されなかった
    housing_median_ageを「中央値補完→標準化」で処理する。
    """

    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)
    default_num_pipeline = build_num_pipeline()
    log_pipeline = build_log_pipeline()

    # ColumnTransformerの出力列名は、常に
    #   "<タプルで指定した名前>__<サブパイプラインのget_feature_names_out()が返す名前>"
    # という形になる。「ratioが後ろ／logが前」に見えるのは、以下のように
    # サブパイプライン側（後半）が返す名前が異なるため。
    #   - ratio_pipeline(): FunctionTransformer(feature_names_out=ratio_name)が
    #     入力列に関係なく常に["ratio"]を返す設計。そのため
    #     "bedrooms__ratio" のように、タプル名(前半)__ratio(固定・後半) になる。
    #   - log_pipeline: FunctionTransformer(feature_names_out="one-to-one")が
    #     入力列名をそのまま返す（例: "total_bedrooms"）。そのため
    #     "log__total_bedrooms" のように、log(タプル名・前半)__元の列名(後半) になる。
    # どちらも「前半=タプル名、後半=サブパイプラインの出力名」という同じ規則で、
    # 後半に来る名前の中身（固定文字列 vs 元列名）が違うだけ。
    #
    # 変換に使われた元の列は、出力には残らない（変換後の列名に置き換わる）。
    #   - 例えば"total_rooms"は bedrooms / rooms_per_house / log の3つのタプルで
    #     入力として重複して使われているが、これは「同じ元列を複数の変換器に
    #     渡して構わない」という話であり、それぞれの出力（"bedrooms__ratio"
    #     "rooms_per_house__ratio" "log__total_rooms"）が別々に追加されるだけで、
    #     元の"total_rooms"列そのものが無変換のまま出力に残るわけではない。
    #   - remainder=default_num_pipeline とした housing_median_age も同様。
    #     "remainder__housing_median_age"という列名で出力されるが、中身は
    #     default_num_pipeline（中央値補完→標準化）を通した後の値であり、
    #     元のスケールの値がそのまま残るわけではない。
    #   - 元の値をそのまま（無変換で）残したい場合は、remainder="passthrough"や、
    #     タプルの変換器に"passthrough"を指定する必要がある。
    return ColumnTransformer(
        [
            # -> "bedrooms__ratio"
            ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            # -> "rooms_per_house__ratio"
            ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
            # -> "people_per_house__ratio"
            ("people_per_house", ratio_pipeline(), ["population", "households"]),
            (
                # -> "log__total_bedrooms", "log__total_rooms", "log__population",
                #    "log__households", "log__median_income"
                "log",
                log_pipeline,
                ["total_bedrooms", "total_rooms", "population", "households",
                 "median_income"],
            ),
            # -> "geo__Cluster 0 similarity" 〜 "geo__Cluster 9 similarity"
            # （ClusterSimilarity.get_feature_names_out()がその名前を返すため）
            ("geo", cluster_simil, ["latitude", "longitude"]),
            # -> "cat__ocean_proximity_<1H OCEAN>" など
            # （OneHotEncoderがカテゴリ値ごとの列名を返すため）
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ],
        # 上記どれにも該当しない列（housing_median_age）はdefault_num_pipelineで処理され、
        # "remainder__housing_median_age" という名前になる。
        remainder=default_num_pipeline,
    )


def save_pipeline_diagram(estimator, file_name: str) -> Path:
    """set_config(display='diagram')のHTML表現をファイルへ保存する。"""

    set_config(display="diagram")
    path = output_path(file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(estimator._repr_html_(), encoding="utf-8")
    return path


def main() -> None:
    setup_logger(Path(__file__).stem)
    housing = load_features()
    housing_num = housing[NUM_ATTRIBS]
    logger.info(f"訓練セット（特徴量）: {housing.shape[0]:,}行 x {housing.shape[1]}列")

    logger.info("\n=== 数値パイプライン（中央値補完→標準化） ===")
    num_pipeline = build_num_pipeline()
    housing_num_prepared = num_pipeline.fit_transform(housing_num)
    logger.info(f"変換後shape: {housing_num_prepared.shape}")
    logger.info(f"先頭2行（丸め）:\n{np.round(housing_num_prepared[:2], 2)}")
    df_prepared = pd.DataFrame(
        housing_num_prepared,
        columns=num_pipeline.get_feature_names_out(),
        index=housing_num.index,
    )
    logger.info(f"get_feature_names_out(): {list(df_prepared.columns)}")

    logger.info("\n=== Pipelineの操作（steps / インデックス / named_steps） ===")
    inspection = inspect_pipeline(num_pipeline)
    logger.info(f"steps: {inspection['steps']}")
    logger.info(f"pipeline[1]（2番目のステップ）: {inspection['second_step']}")
    logger.info(f"pipeline[:-1]（最後を除くステップ名）: {inspection['all_but_last']}")
    logger.info(f"named_steps['simpleimputer']: {inspection['named_step']}")
    num_pipeline.set_params(simpleimputer__strategy="median")
    logger.info(
        "set_paramsでstrategyを明示的に上書きできる（探索用にハイパーパラメータ名を"
        "ステップ名__パラメータ名の形式で指定する）。"
    )

    logger.info("\n=== ColumnTransformer: 列名指定 vs dtype選択 ===")
    cat_pipeline = build_cat_pipeline()
    by_name = build_column_transformer_by_name(build_num_pipeline(), cat_pipeline)
    by_dtype = build_column_transformer_by_dtype(build_num_pipeline(), build_cat_pipeline())
    prepared_by_name = by_name.fit_transform(housing)
    prepared_by_dtype = by_dtype.fit_transform(housing)
    logger.info(f"列名指定の出力shape: {prepared_by_name.shape}")
    logger.info(f"dtype選択の出力shape: {prepared_by_dtype.shape}")
    logger.info(
        f"両者が一致するか: {np.allclose(prepared_by_name, prepared_by_dtype)}"
    )
    logger.info(
        "make_column_selector(dtype_include=...)を使うと、列名を書き並べずに"
        "型（数値/objectなど）でnum_attribs・cat_attribsに相当する列集合を選べる。"
    )

    logger.info("\n=== 比率・対数・クラスタ類似度・カテゴリを統合したColumnTransformer ===")
    full_preprocessing = build_full_preprocessing(build_cat_pipeline())
    housing_prepared = full_preprocessing.fit_transform(housing)
    feature_names = full_preprocessing.get_feature_names_out()
    logger.info(f"統合後の出力shape: {housing_prepared.shape}")
    logger.info(f"出力列数: {len(feature_names)}")
    logger.info(f"出力列名: {list(feature_names[:])}")
    logger.info(
        "remainderに渡したdefault_num_pipelineが、比率・対数・地理類似度・"
        "カテゴリ変換のどれにも使われなかったhousing_median_ageを処理する。"
        "個別の変換パイプラインを1つのColumnTransformerへまとめることで、"
        "学習・交差検証・推論のいずれでも同じ前処理が再現される。"
    )

    logger.info("\n=== パイプライン構成の可視化（set_config(display='diagram')） ===")
    num_diagram_path = save_pipeline_diagram(num_pipeline, "num_pipeline_diagram.html")
    full_diagram_path = save_pipeline_diagram(
        full_preprocessing, "full_preprocessing_diagram.html"
    )
    logger.info(f"保存しました: {num_diagram_path}")
    logger.info(f"保存しました: {full_diagram_path}")


if __name__ == "__main__":
    main()
