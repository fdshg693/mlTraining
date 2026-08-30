"""Download a Kaggle dataset or competition dataset via kagglehub.

Downloads the specified Kaggle dataset (or competition dataset) using
``kagglehub`` and copies the files into ``kaggle/data/{dataset-name}/``.

Kaggle認証が必要です。事前に ``~/.kaggle/kaggle.json`` を配置するか、
環境変数 ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` を設定してください。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub
import typer

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "kaggle" / "data"

app = typer.Typer(add_completion=False)


@app.command()
def main(
    slug: str = typer.Argument(
        ..., help="Kaggleのデータセット識別子 (例: titanic, owner/dataset-name)"
    ),
    competition: bool = typer.Option(
        False,
        "--competition/--dataset",
        help="コンペティションのデータをダウンロードする場合に指定 (デフォルトは通常のデータセット)",
    ),
    name: str = typer.Option(
        None,
        "--name",
        help="配置先フォルダ名 (省略時はslugの末尾から自動生成)",
    ),
) -> None:
    dataset_name = name or slug.rstrip("/").split("/")[-1]
    dest_dir = OUTPUT_ROOT / dataset_name

    if competition:
        source_dir = Path(kagglehub.competition_download(slug))
    else:
        source_dir = Path(kagglehub.dataset_download(slug))

    dest_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        target = dest_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    print(f"downloaded {slug} -> {dest_dir}")


if __name__ == "__main__":
    app()
