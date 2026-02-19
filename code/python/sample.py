# `data/input/train.csv` のデータを元に主な統計値などをCSV形式で出力するサンプルコードです。

import pandas as pd
from path_utils import input_path, output_path

# パス設定
INPUT_PATH = input_path("train.csv")
OUTPUT_DIR_PATH = output_path("sample")

OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)

# データ読み込み
df = pd.read_csv(INPUT_PATH)

# --- 1. 基本統計量 (describe) ---
desc = df.describe(include="all").T
desc.index.name = "column"
desc.to_csv(OUTPUT_DIR_PATH / "basic_statistics.csv")
print("基本統計量を出力しました。")

# --- 2. 欠損値の集計 ---
missing = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_rate": df.isnull().mean().round(4),
})
missing.index.name = "column"
missing.to_csv(OUTPUT_DIR_PATH / "missing_values.csv")
print("欠損値集計を出力しました。")

# --- 3. 生存率のクロス集計 (性別 × クラス) ---
survival = df.groupby(["Sex", "Pclass"])["Survived"].agg(["count", "sum", "mean"])
survival.columns = ["total", "survived", "survival_rate"]
survival["survival_rate"] = survival["survival_rate"].round(4)
survival.to_csv(OUTPUT_DIR_PATH / "survival_by_sex_class.csv")
print("生存率クロス集計を出力しました。")

# --- 4. 年齢分布 (ヒストグラム用のビン集計) ---
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
labels = [f"{b}-{b+10}" for b in bins[:-1]]
df["AgeBin"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
age_dist = df.groupby("AgeBin", observed=False)["Survived"].agg(["count", "sum", "mean"])
age_dist.columns = ["total", "survived", "survival_rate"]
age_dist["survival_rate"] = age_dist["survival_rate"].round(4)
age_dist.to_csv(OUTPUT_DIR_PATH / "age_distribution.csv")
print("年齢分布を出力しました。")

# --- 5. 数値列の相関行列 ---
corr = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].corr().round(4)
corr.index.name = "column"
corr.to_csv(OUTPUT_DIR_PATH / "correlation_matrix.csv")
print("相関行列を出力しました。")

print("\nすべての出力が完了しました。出力先: data/output/")
