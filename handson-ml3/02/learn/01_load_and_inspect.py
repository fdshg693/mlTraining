"""住宅価格データを取得し、データの構造を確認する。"""

import io
from pathlib import Path
import sys
import tarfile
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_URL = "https://github.com/ageron/data/raw/main/housing.tgz"
HOUSING_TARBALL = "housing.tgz"
HOUSING_CSV = "housing/housing.csv"

sys.path.insert(0, str(PROJECT_DIR))
from data_produce.util.file_io import data_path, read_csv, save_figure
from data_produce.util.logging_config import setup_logger
from data_produce.util.metadata import record_write


def load_housing_data(
    data_dir: str | Path | None = None,
    url: str = DATA_URL,
) -> pd.DataFrame:
    """住宅価格データをローカルに用意してDataFrameとして返す。

    tarballと展開済みCSVを別々に確認することで、再実行時には不要な
    ダウンロード・展開を行わない。``data_dir``を変更すれば、テスト用の
    一時ディレクトリや別の保存先も利用できる。
    """

    data_dir = (
        data_path(HOUSING_TARBALL).parent
        if data_dir is None
        else Path(data_dir)
    )
    tarball_path = data_path(HOUSING_TARBALL, data_dir)
    housing_csv_path = data_path(HOUSING_CSV, data_dir)

    if not housing_csv_path.is_file():
        tarball_path.parent.mkdir(parents=True, exist_ok=True)

        if not tarball_path.is_file():
            logger.info(f"データをダウンロードします: {url}")
            urllib.request.urlretrieve(url, tarball_path)
            record_write(tarball_path)

        logger.info(f"アーカイブを展開します: {tarball_path}")
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path=tarball_path.parent)
        record_write(housing_csv_path)

    if not housing_csv_path.is_file():
        raise FileNotFoundError(
            f"展開後のCSVが見つかりません: {housing_csv_path}"
        )

    return read_csv(HOUSING_CSV, data_dir)


def inspect_housing_data(
    housing: pd.DataFrame,
    outputs_dir: str | Path | None = None,
) -> Path:
    """基本的なデータ確認結果を表示し、数値列のヒストグラムを保存する。"""

    logger.info("=== 先頭5行 ===")
    logger.info(f"\n{housing.head().to_string(index=False)}")

    info_buffer = io.StringIO()
    housing.info(buf=info_buffer)
    logger.info(f"\n=== データ構造 ===\n{info_buffer.getvalue()}")

    logger.info(
        "\n=== ocean_proximity の値の数 ===\n"
        f"{housing['ocean_proximity'].value_counts(dropna=False).to_string()}"
    )

    logger.info(f"\n=== 数値列の要約統計量 ===\n{housing.describe().to_string()}")

    numeric_housing = housing.select_dtypes(include="number")
    axes = numeric_housing.hist(bins=50, figsize=(12, 8))  # bins=50: 各ヒストグラムの棒(区間)の数
    first_axis = axes.flat[0] if hasattr(axes, "flat") else axes
    figure = first_axis.get_figure()
    figure.tight_layout()

    if outputs_dir is None:
        histogram_path = save_figure(
            figure, "attribute_histogram_plots.png", dpi=300
        )
    else:
        histogram_path = save_figure(
            figure,
            "attribute_histogram_plots.png",
            outputs_dir,
            dpi=300,
        )
    plt.close(figure)
    logger.info(f"\nヒストグラムを保存しました: {histogram_path}")
    return histogram_path


def main() -> None:
    setup_logger(Path(__file__).stem)
    housing = load_housing_data()
    logger.info(f"データセット: {housing.shape[0]:,}行 x {housing.shape[1]}列")
    inspect_housing_data(housing)


if __name__ == "__main__":
    main()
