"""CSVの読み書きを `data_produce/data/` 配下に集約する薄いラッパー。

呼び出し側はCSVファイル名だけを指定すればよく、`data_produce/data` への
相対パス（フォルダ階層）を意識する必要がない。
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def data_path(csv_name: str) -> Path:
    """`data_produce/data/{csv_name}` の絶対パスを返す"""
    return DATA_DIR / csv_name


def exists(csv_name: str) -> bool:
    """`data_produce/data/{csv_name}` が既に存在するか"""
    return data_path(csv_name).is_file()


def read_csv(csv_name: str, **kwargs) -> pd.DataFrame:
    """`data_produce/data/{csv_name}` を読み込む（kwargsはpd.read_csvにそのまま渡す）"""
    return pd.read_csv(data_path(csv_name), **kwargs)


def write_csv(df: pd.DataFrame, csv_name: str, **kwargs) -> Path:
    """DataFrameを`data_produce/data/{csv_name}`へ書き出す（kwargsはDataFrame.to_csvにそのまま渡す）"""
    path = data_path(csv_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **kwargs)
    return path
