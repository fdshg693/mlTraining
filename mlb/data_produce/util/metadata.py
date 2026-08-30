"""データ・生成物ファイルの作成元/利用先を記録するレジストリ。

read_csv/write_csv/save_figureなどが呼ばれるたびに、対象ファイルと
呼び出し元スクリプトの対応をdata_registry.jsonへ追記・更新する。
ソースコードを一つずつ読まなくても、どのファイルがどのスクリプトで
作られ、どこで読まれているかを一覧できるようにするための仕組み。
"""

import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_DIR / "data_registry.json"

_UTIL_DIR = Path(__file__).resolve().parent


def _relative_to_project(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def _caller_relpath() -> str:
    """util配下のラッパー関数を飛ばして、直接の呼び出し元スクリプトを特定する。"""

    for frame_info in inspect.stack()[1:]:
        frame_path = Path(frame_info.filename).resolve()
        if frame_path.parent != _UTIL_DIR:
            return _relative_to_project(frame_path)
    return "unknown"


def _load_registry() -> dict:
    if REGISTRY_PATH.is_file():
        with REGISTRY_PATH.open(encoding="utf-8") as registry_file:
            return json.load(registry_file)
    return {}


def _save_registry(registry: dict) -> None:
    with REGISTRY_PATH.open("w", encoding="utf-8") as registry_file:
        json.dump(registry, registry_file, ensure_ascii=False, indent=2, sort_keys=True)
        registry_file.write("\n")


def _record(path: Path, bucket_name: str) -> None:
    key = _relative_to_project(path)
    caller = _caller_relpath()

    registry = _load_registry()
    entry = registry.setdefault(key, {"writers": [], "readers": []})
    bucket = entry.setdefault(bucket_name, [])
    if caller not in bucket:
        bucket.append(caller)
        bucket.sort()
    entry["last_updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _save_registry(registry)


def record_write(path: Path) -> None:
    """pathを書き込んだ呼び出し元スクリプトを記録する。"""

    _record(path, "writers")


def record_read(path: Path) -> None:
    """pathを読み込んだ呼び出し元スクリプトを記録する。"""

    _record(path, "readers")
