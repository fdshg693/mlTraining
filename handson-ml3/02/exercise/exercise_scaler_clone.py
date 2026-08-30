"""06_custom_transformers.pyのStandardScalerCloneを完成させる（演習6）。

06では「fit/transformの契約」「check_is_fittedによるfit前呼び出しの検出」だけを
確認していた。ここでは以下を追加し、sklearn.preprocessing.StandardScalerに
近い挙動へ仕上げる。

- with_mean=Falseでも動く標準化（中心化せず、スケールだけ揃える）。
- 標準偏差が0の定数列でゼロ除算・NaNが出ないようにする。
- inverse_transform()で変換前の値へ戻せること。
- DataFrame入力時にfeature_names_in_を記憶し、get_feature_names_out()で
  列名（またはデフォルトのx0, x1, ...形式）を返し、fit時と異なる列名・列数の
  入力を検出する。
- check_estimator()でscikit-learnの推定器APIに従えていることを確認する。

check_array + 手作業の列数チェックだった06版の代わりに、
sklearn.utils.validation.validate_data（現行scikit-learnの検証関数）を使う。
validate_data()はfit時にn_features_in_・feature_names_in_を自動で記録し、
transform側（reset=False）で列数・列名の整合性を検証してくれるため、
手動チェックより少ないコードでcheck_estimator()の要求を満たせる。
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted, validate_data

PROJECT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "learn"))
from common import load_features_and_labels  # noqa: E402
from data_produce.util.logging_config import setup_logger  # noqa: E402


class StandardScalerClone(TransformerMixin, BaseEstimator):
    """StandardScalerの契約を確認するための再実装（完成版）。

    mixinの継承順はTransformerMixinを先に置く。BaseEstimatorを先に置くと
    check_estimator()のcheck_mixin_orderが「より特化したmixinを左側に」という
    規約違反として検出する。
    """

    def __init__(self, with_mean: bool = True):
        self.with_mean = with_mean

    def fit(self, X, y=None):
        X = validate_data(self, X)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0  # 定数列でのゼロ除算を防ぐ
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = X * self.scale_
        return X + self.mean_ if self.with_mean else X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        if input_features is None:
            if hasattr(self, "feature_names_in_"):
                return np.asarray(self.feature_names_in_, dtype=object)
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)

        input_features = np.asarray(input_features, dtype=object)
        if len(input_features) != self.n_features_in_:
            raise ValueError(
                f"input_featuresの数{len(input_features)}が"
                f"fit時の列数{self.n_features_in_}と一致しません。"
            )
        if hasattr(self, "feature_names_in_") and not np.array_equal(
            input_features, self.feature_names_in_
        ):
            raise ValueError("input_featuresがfit時のfeature_names_in_と一致しません。")
        return input_features


def load_numeric_training_data() -> pd.DataFrame:
    """02_split_data.pyが保存した訓練セットから、欠損補完済みの数値列を返す。

    StandardScalerClone・StandardScalerともに欠損値を扱えないため、
    06_custom_transformers.pyと同様にtotal_bedroomsを中央値で補完する。
    set_output(transform="pandas")で、列名を保ったままDataFrameとして返す。
    """

    housing, _housing_labels = load_features_and_labels()
    housing_num = housing.select_dtypes(include=[np.number])
    imputer = SimpleImputer(strategy="median").set_output(transform="pandas")
    return imputer.fit_transform(housing_num)


def compare_with_sklearn(
    housing_num: pd.DataFrame, with_mean: bool
) -> tuple[np.ndarray, np.ndarray]:
    """StandardScalerCloneとStandardScalerの出力をwith_mean別に比較する。"""

    clone_scaled = StandardScalerClone(with_mean=with_mean).fit_transform(housing_num)
    sklearn_scaled = StandardScaler(with_mean=with_mean).fit_transform(housing_num)
    return clone_scaled, sklearn_scaled


def check_inverse_transform(housing_num: pd.DataFrame, with_mean: bool) -> bool:
    """transform()した結果をinverse_transform()で戻し、元の値と一致するか確認する。"""

    scaler = StandardScalerClone(with_mean=with_mean).fit(housing_num)
    scaled = scaler.transform(housing_num)
    restored = scaler.inverse_transform(scaled)
    return np.allclose(restored, housing_num.to_numpy())


def demo_constant_column() -> np.ndarray:
    """定数列を含むデータをfitし、scale_が0にならないことを確認する。"""

    rng = np.random.default_rng(42)
    X = np.column_stack([rng.normal(size=20), np.full(20, 7.0)])
    scaler = StandardScalerClone().fit(X)
    transformed = scaler.transform(X)
    if np.isnan(transformed).any() or np.isinf(transformed).any():
        raise AssertionError("定数列の標準化でNaN/Infが発生しました。")
    return scaler.scale_


def demo_feature_names(housing_num: pd.DataFrame) -> dict[str, object]:
    """DataFrame/ndarray入力での列名の扱いと、不一致検出を確認する。"""

    scaler_from_df = StandardScalerClone().fit(housing_num)
    scaler_from_array = StandardScalerClone().fit(housing_num.to_numpy())

    renamed = housing_num.rename(columns={housing_num.columns[0]: "renamed_column"})
    try:
        scaler_from_df.transform(renamed)
        mismatch_detected = False
    except ValueError:
        mismatch_detected = True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scaler_from_df.transform(housing_num.to_numpy())
        warned_on_missing_names = any(
            issubclass(w.category, UserWarning) for w in caught
        )

    return {
        "feature_names_in_": list(scaler_from_df.feature_names_in_),
        "get_feature_names_out_from_df": list(scaler_from_df.get_feature_names_out()),
        "get_feature_names_out_from_array": list(scaler_from_array.get_feature_names_out()),
        "mismatch_detected": mismatch_detected,
        "warned_on_missing_names": warned_on_missing_names,
    }


def main() -> None:
    setup_logger(Path(__file__).stem)
    housing_num = load_numeric_training_data()
    logger.info(f"訓練セット（数値列、欠損補完済み）: {housing_num.shape[0]:,}行 x {housing_num.shape[1]}列")

    logger.info("\n=== check_estimatorでscikit-learn APIへの適合を確認 ===")
    check_estimator(StandardScalerClone())
    check_estimator(StandardScalerClone(with_mean=False))
    logger.info(
        "with_mean=True/Falseの両方でエラーなし。"
        "validate_data()がfit/transform間のn_features_in_・feature_names_in_の"
        "整合性チェックを肩代わりしてくれるため、06版のcheck_array単独実装より"
        "少ないコードでAPI契約を満たせる。"
    )

    logger.info("\n=== with_mean別にStandardScalerと出力を比較 ===")
    for with_mean in (True, False):
        clone_scaled, sklearn_scaled = compare_with_sklearn(housing_num, with_mean)
        all_close = np.allclose(clone_scaled, sklearn_scaled)
        logger.info(f"with_mean={with_mean}: 両者の出力が一致するか: {all_close}")

    logger.info("\n=== inverse_transform()で変換前へ戻せるか ===")
    for with_mean in (True, False):
        restored_ok = check_inverse_transform(housing_num, with_mean)
        logger.info(f"with_mean={with_mean}: inverse_transform()が元の値へ戻るか: {restored_ok}")

    logger.info("\n=== 定数列でのゼロ除算対策 ===")
    scale_with_constant = demo_constant_column()
    logger.info(f"scale_（2列目が定数列）: {scale_with_constant.round(3)}")
    logger.info(
        "定数列の標準偏差は0だが、scale_[scale_ == 0] = 1としているためゼロ除算せず、"
        "定数列の変換結果は「元の値 - 平均」（with_mean=Trueなら0）のまま出力される。"
    )

    logger.info("\n=== DataFrame入力時の列名の扱い ===")
    feature_name_result = demo_feature_names(housing_num)
    logger.info(f"feature_names_in_: {feature_name_result['feature_names_in_']}")
    logger.info(
        "get_feature_names_out()（DataFrameでfit）: "
        f"{feature_name_result['get_feature_names_out_from_df']}"
    )
    logger.info(
        "get_feature_names_out()（ndarrayでfit、デフォルトx0形式）: "
        f"{feature_name_result['get_feature_names_out_from_array']}"
    )
    logger.info(
        f"fit時と異なる列名のDataFrameをtransform()すると例外になるか: "
        f"{feature_name_result['mismatch_detected']}"
    )
    logger.info(
        "fit時はDataFrame・transform時はndarray（列名なし）だと警告が出るか: "
        f"{feature_name_result['warned_on_missing_names']}"
    )


if __name__ == "__main__":
    main()
