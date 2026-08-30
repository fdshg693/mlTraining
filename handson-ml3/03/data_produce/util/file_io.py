"""データセットと生成物のファイル入出力を集約する薄いラッパー。

呼び出し側はプロジェクトルートからの相対パスや出力先の作り方を意識せず、
ファイル名だけを指定して読み書きできる。outputs_dirを省略した場合は、
呼び出し元スクリプトと同じディレクトリのoutputsフォルダが使われる
（learn配下から呼べばlearn/outputs、exercise配下から呼べばexercise/outputs）。
"""

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_produce.util.metadata import record_read, record_write


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _caller_outputs_dir() -> Path:
    """このヘルパーを直接呼んだ関数の、そのまた呼び出し元スクリプトのディレクトリ
    直下のoutputsフォルダを返す（output_path/save_figureから直接呼ぶこと）。
    """

    caller_file = Path(inspect.stack()[2].filename).resolve()
    return caller_file.parent / "outputs"


def data_path(file_name: str, data_dir: str | Path = DATA_DIR) -> Path:
    """データファイルの絶対パスを返す。"""

    return Path(data_dir) / file_name


def output_path(file_name: str, outputs_dir: str | Path | None = None) -> Path:
    """スクリプトの生成物の絶対パスを返す。"""

    if outputs_dir is None:
        outputs_dir = _caller_outputs_dir()
    return Path(outputs_dir) / file_name


def exists(file_name: str, data_dir: str | Path = DATA_DIR) -> bool:
    """データファイルが既に存在するか確認する。"""

    return data_path(file_name, data_dir).is_file()


def read_csv(
    csv_name: str,
    data_dir: str | Path = DATA_DIR,
    **kwargs: Any,
) -> pd.DataFrame:
    """データディレクトリ配下のCSVを読み込む。"""

    path = data_path(csv_name, data_dir)
    record_read(path)
    return pd.read_csv(path, **kwargs)


def write_csv(
    df: pd.DataFrame,
    csv_name: str,
    data_dir: str | Path = DATA_DIR,
    **kwargs: Any,
) -> Path:
    """DataFrameをデータディレクトリ配下のCSVへ書き出す。"""

    path = data_path(csv_name, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **kwargs)
    record_write(path)
    return path


def read_npz(
    npz_name: str,
    data_dir: str | Path = DATA_DIR,
) -> dict[str, np.ndarray]:
    """データディレクトリ配下のnpzを読み込み、配列名をキーにした辞書として返す。"""

    path = data_path(npz_name, data_dir)
    record_read(path)
    with np.load(path) as npz_file:
        return {key: npz_file[key] for key in npz_file.files}


def write_npz(
    arrays: dict[str, np.ndarray],
    npz_name: str,
    data_dir: str | Path = DATA_DIR,
) -> Path:
    """複数のndarrayをデータディレクトリ配下のnpzへ書き出す。"""

    path = data_path(npz_name, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    record_write(path)
    return path


def save_figure(
    figure: Any,
    file_name: str,
    outputs_dir: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    """Figureを出力ディレクトリへ保存する。"""

    if outputs_dir is None:
        outputs_dir = _caller_outputs_dir()
    path = output_path(file_name, outputs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, **kwargs)
    record_write(path)
    return path
