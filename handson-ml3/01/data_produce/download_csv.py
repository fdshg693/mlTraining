# lifesat.csv（閾値内データ）をそのままダウンロードして data/ に配置するスクリプト

from urllib.request import urlretrieve

from pathlib import Path

from util.csv_io import data_path
from util.logging_config import setup_logger

CSV_URL = "https://github.com/ageron/data/raw/main/lifesat/lifesat.csv"
CSV_NAME = "lifesat.csv"


def main() -> None:
    logger = setup_logger(Path(__file__).stem)
    path = data_path(CSV_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(CSV_URL, path)
    logger.info(f"Downloaded {CSV_URL} to {path}")


if __name__ == "__main__":
    main()
