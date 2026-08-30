"""baseballsavant.mlb.comのカスタムカラム選択UIから、選択可能なスタッツ一覧を取得する。

`fetch_leaderboard.py`の`--field`（`savant_client.SavantField`）に指定できる値の
候補を網羅的に把握するための下調べ用スクリプト。カスタムリーダーボード画面
（`.../leaderboard/custom?...`）のHTMLには、選択可能なスタッツが
`<input type="checkbox" id="p_game" ...>`のようなチェックボックスとして
列挙されており、その`id`がそのまま`selections`パラメータの値になる。

このHTMLは過去に使われていたスタッツ定義がHTMLコメント（`<!-- ... -->`）として
そのまま残っており、正規表現で「見出しから次の見出しまで」を素朴に抜き出すと
コメント内の重複idまで拾ってしまう。BeautifulSoupでツリーとして解析すれば
コメントはテキストとして扱われ、実際に有効なチェックボックスだけが残る。
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import requests
from bs4 import BeautifulSoup
from loguru import logger

from data_produce.util.file_io import data_path, write_yaml
from data_produce.util.metadata import record_write


BASE_URL = "https://baseballsavant.mlb.com/leaderboard/custom"

# カスタムカラム選択UI上の見出し（h6のテキスト）→ YAML内のカテゴリキー。
# "Statcast Stats"の親div.flexは"Bat Tracking"等の下位見出しとも箱を共有しているため、
# statcast_stats配下にはそれらのスタッツも一緒に含まれる。
CATEGORY_HEADINGS = {
    "standard_stats": "Standard Stats",
    "statcast_stats": "Statcast Stats",
}


def _extract_category(soup: BeautifulSoup, heading_text: str) -> list[dict[str, str | None]]:
    """指定した見出し（h6）の親`div.flex`配下にあるチェックボックスをid/labelの一覧にする。"""

    heading = soup.find("h6", string=heading_text)
    if heading is None:
        raise ValueError(f"見出しが見つかりません: {heading_text}")

    container = heading.find_parent("div", class_="flex")
    if container is None:
        raise ValueError(f"'{heading_text}'の親div.flexが見つかりません")

    stats = []
    for checkbox in container.find_all("input", {"type": "checkbox"}):
        # label の for 属性はid とタイポでずれている場合がある（例: arm_angle）ため、
        # for属性ではなくDOM上の直後の<label>を採る。
        label = checkbox.find_next_sibling("label")
        stats.append(
            {
                "id": checkbox.get("id"),
                "label": label.get_text(strip=True) if label else None,
            }
        )
    return stats


def fetch_stat_catalog(
    year: int,
    player_type: Literal["pitcher", "batter"],
    refresh: bool = False,
) -> Path:
    """指定した年・対象のカスタムカラム選択UIから、選択可能なスタッツ一覧をYAMLに保存する。

    既存ファイルがあり`refresh=False`（デフォルト）ならそのまま返す。
    `refresh=True`か未取得の場合のみ取得し直し、取得条件を
    `stat_catalog.meta.json`へ書き出す（キャッシュヒット時はmeta.jsonを更新しない）。
    """

    yaml_path = data_path(f"stat_catalog/{player_type}/{year}/stat_catalog.yaml")
    meta_path = yaml_path.with_suffix(".meta.json")

    if yaml_path.is_file() and not refresh:
        logger.info(f"既存のYAMLを使用します（再取得スキップ）: {yaml_path}")
        return yaml_path

    params = {
        "year": year,
        "type": player_type,
        "filter": "",
        "min": "q",
        "selections": "",
        "chart": "false",
        "x": "",
        "y": "",
        "r": "no",
        "chartType": "beeswarm",
        "sort": "xwoba",
        "sortDir": "asc",
    }

    logger.info(f"カスタムカラム選択画面を取得します: {BASE_URL} params={params}")
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    categories = {
        category_key: _extract_category(soup, heading_text)
        for category_key, heading_text in CATEGORY_HEADINGS.items()
    }

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    content = {
        "year": year,
        "player_type": player_type,
        "fetched_at": fetched_at,
        "url": response.url,
        "categories": categories,
    }

    relative_yaml_name = f"stat_catalog/{player_type}/{year}/stat_catalog.yaml"
    write_yaml(content, relative_yaml_name)

    meta = {
        "url": response.url,
        "params": params,
        "fetched_at": fetched_at,
    }
    with meta_path.open("w", encoding="utf-8") as meta_file:
        json.dump(meta, meta_file, ensure_ascii=False, indent=2)
        meta_file.write("\n")
    record_write(meta_path)

    return yaml_path
