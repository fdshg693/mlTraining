"""年・対象（投手/打者）・フィールドを指定してbaseballsavantのリーダーボードCSVを取得するCLI。"""

from pathlib import Path
import sys

import typer

PROJECT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_DIR))
from data_produce import savant_client
from data_produce.util.file_io import read_csv
from data_produce.util.logging_config import setup_logger


app = typer.Typer(help="baseballsavantリーダーボードCSVの取得CLI", add_completion=False)


@app.command()
def fetch(
    year: int = typer.Option(2026, "--year", "-y", help="取得対象の年"),
    player_type: str = typer.Option(
        "pitcher", "--type", "-t", help="取得対象（pitcher または batter）"
    ),
    field: list[savant_client.SavantField] = typer.Option(
        [savant_client.SavantField.FASTBALL_VELOCITY],
        "--field",
        "-f",
        help="取得するフィールド（複数指定可。組み合わせに応じたファイル名を自動生成する）",
    ),
    min_sample: savant_client.MinSample = typer.Option(
        savant_client.MinSample.QUALIFIED,
        "--min-sample",
        help="最小サンプル数（qで規定投球回/打席数）",
    ),
    refresh: bool = typer.Option(False, "--refresh", help="既存CSVがあっても再取得する"),
) -> None:
    """指定した年・対象・フィールドの組み合わせのリーダーボードCSVを取得し、内容をログへ出力する。"""

    if player_type not in ("pitcher", "batter"):
        raise typer.BadParameter("typeは'pitcher'か'batter'を指定してください")

    logger = setup_logger(Path(__file__).stem)

    csv_path = savant_client.fetch_leaderboard(
        year=year,
        player_type=player_type,
        fields=field,
        min_sample=min_sample,
        refresh=refresh,
    )

    stat_key = savant_client.stat_key_for(field)
    df = read_csv(f"leaderboard/{player_type}/{year}/{stat_key}.csv")
    logger.info(f"[{stat_key}] 保存先: {csv_path} / {df.shape[0]:,}行 x {df.shape[1]}列")
    logger.info(f"[{stat_key}] 列名: {list(df.columns)}")


if __name__ == "__main__":
    app()
