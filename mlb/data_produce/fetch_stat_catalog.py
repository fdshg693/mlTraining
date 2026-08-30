"""年・対象（投手/打者）を指定して、baseballsavantで選択可能なスタッツ一覧を取得するCLI。"""

from pathlib import Path
import sys

import typer

PROJECT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_DIR))
from data_produce import stat_catalog
from data_produce.util.file_io import read_yaml
from data_produce.util.logging_config import setup_logger


app = typer.Typer(help="baseballsavantスタッツ一覧の取得CLI", add_completion=False)


@app.command()
def fetch(
    year: int = typer.Option(2026, "--year", "-y", help="取得対象の年"),
    player_type: str = typer.Option(
        "pitcher", "--type", "-t", help="取得対象（pitcher または batter）"
    ),
    refresh: bool = typer.Option(False, "--refresh", help="既存YAMLがあっても再取得する"),
) -> None:
    """指定した年・対象で選択可能なスタッツ一覧を取得し、カテゴリごとの件数をログへ出力する。"""

    if player_type not in ("pitcher", "batter"):
        raise typer.BadParameter("typeは'pitcher'か'batter'を指定してください")

    logger = setup_logger(Path(__file__).stem)

    yaml_path = stat_catalog.fetch_stat_catalog(year=year, player_type=player_type, refresh=refresh)

    content = read_yaml(f"stat_catalog/{player_type}/{year}/stat_catalog.yaml")
    for category_key, stats in content["categories"].items():
        ids = [stat["id"] for stat in stats]
        logger.info(f"[{category_key}] {len(ids)}件: {ids}")
    logger.info(f"保存先: {yaml_path}")


if __name__ == "__main__":
    app()
