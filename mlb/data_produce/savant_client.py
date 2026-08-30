"""baseballsavant.mlb.comのリーダーボードCSVを取得する。

シーズン進行中は値が更新され続けるため、02のhousing.csvのような
「一度取得すれば終わり」の静的データとは前提が異なる。既存ファイルが
あればデフォルトでは再取得せず使い回しつつ、`refresh=True`で明示的に
再取得できるようにし、取得条件（URL・パラメータ・取得日時）は必ず
meta.jsonへ残す。
"""

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal, Sequence

import requests
from loguru import logger

from data_produce.util.file_io import data_path
from data_produce.util.metadata import record_write


BASE_URL = "https://baseballsavant.mlb.com/leaderboard/custom"


class SavantField(str, Enum):
    """baseballsavantのselectionsに指定できるフィールド。

    ダウンロード時に指定できる列はここで管理する。新しいフィールドが
    必要になったらメンバーを追記するだけでよい（組み合わせごとの登録は不要）。
    """

    FASTBALL_VELOCITY = "ff_avg_speed"
    FASTBALL_SPIN = "ff_avg_spin"
    INNINGS_PITCHED = "p_formatted_ip"
    GAMES = "p_game"
    FASTBALL_USAGE = "n_ff_formatted"
    SINKER_USAGE = "n_si_formatted"
    SLIDER_USAGE = "n_sl_formatted"
    CURVEBALL_USAGE = "n_cu_formatted"
    CHANGEUP_USAGE = "n_ch_formatted"
    CUTTER_USAGE = "n_fc_formatted"
    ERA = "p_era"
    OPPONENT_BATTING_AVG = "p_opp_batting_avg"


class MinSample(str, Enum):
    """baseballsavantの`min`（最小サンプル数）に指定できる値。

    "q"は規定投球回/打席数で、一年通して先発格として投げた（打った）ことの
    証明にあたる。それ以外の数値は「Nイニング（打席）以上」を意味し、
    savant側で選べる組み合わせが決まっているため任意の数値は指定できない。
    """

    QUALIFIED = "q"
    INNINGS_1 = "1"
    INNINGS_10 = "10"
    INNINGS_20 = "20"
    INNINGS_40 = "40"
    INNINGS_100 = "100"


def stat_key_for(fields: Sequence[SavantField]) -> str:
    """フィールドの組み合わせから、ファイル名に使う一意なキーを生成する。

    指定順によらず同じ組み合わせなら同じキーになるよう、値でソートしてから連結する。
    """

    if not fields:
        raise ValueError("fieldsは1つ以上指定してください")
    return "_".join(sorted(field.value for field in fields))


def fetch_leaderboard(
    year: int,
    player_type: Literal["pitcher", "batter"],
    fields: Sequence[SavantField],
    min_sample: MinSample = MinSample.QUALIFIED,
    extra_params: dict | None = None,
    refresh: bool = False,
) -> Path:
    """指定した年・対象・フィールドの組み合わせのリーダーボードCSVをローカルに用意する。

    ファイル名はフィールドの組み合わせから自動生成する（`stat_key_for`）。
    既存ファイルがあり`refresh=False`（デフォルト）ならそのまま返す。
    `refresh=True`か未取得の場合のみダウンロードし、取得条件を
    `{stat_key}.meta.json`へ書き出す（キャッシュヒット時はmeta.jsonを
    更新しない＝そのCSVが実際に取得された時点の条件を保つ）。
    """

    stat_key = stat_key_for(fields)
    csv_path = data_path(f"leaderboard/{player_type}/{year}/{stat_key}.csv")
    meta_path = csv_path.with_suffix(".meta.json")

    if csv_path.is_file() and not refresh:
        logger.info(f"既存のCSVを使用します（再取得スキップ）: {csv_path}")
        return csv_path

    selections = [field.value for field in fields]
    params = {
        "year": year,
        "type": player_type,
        "filter": "",
        "min": min_sample.value,
        "selections": ",".join(selections),
        "chart": "false",
        "x": selections[0],
        "y": selections[0],
        "r": "no",
        "chartType": "beeswarm",
        "sort": 1,
        "sortDir": "asc",
        "csv": "true",
    }
    if extra_params:
        params.update(extra_params)

    logger.info(f"リーダーボードを取得します: {BASE_URL} params={params}")
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(response.text, encoding="utf-8")
    record_write(csv_path)

    meta = {
        "url": response.url,
        "params": params,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with meta_path.open("w", encoding="utf-8") as meta_file:
        json.dump(meta, meta_file, ensure_ascii=False, indent=2)
        meta_file.write("\n")
    record_write(meta_path)

    return csv_path
