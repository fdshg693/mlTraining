"""呼び出し元スクリプト向けにloguruのログ出力先を設定する。

このモジュール自身はログを出力しない。setup_logger()は呼び出し元スクリプトの
場所から「呼び出し元スクリプトと同じディレクトリのlogsフォルダ」を出力先として
決めるだけを担当する（learn配下から呼べばlearn/logs、experiment配下から呼べば
experiment/logsに出力される）。
"""

import inspect
from datetime import datetime
from pathlib import Path

from loguru import logger

LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"


def setup_logger(script_name: str, mode: str = "overwrite"):
    """スクリプトごとのログファイルを設定し、loguruのloggerを返す。

    mode="overwrite"（デフォルト）: 実行のたびに同名のログファイルを上書きする。
    mode="new": タイムスタンプを先頭に付けた新しいログファイルを毎回作成する。
    """

    caller_dir = Path(inspect.stack()[1].filename).resolve().parent
    logs_dir = caller_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if mode == "new":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"{timestamp}_{script_name}.log"
    elif mode == "overwrite":
        log_path = logs_dir / f"{script_name}.log"
    else:
        raise ValueError(f"未知のmodeです: {mode}（'overwrite'か'new'を指定してください）")

    logger.remove()
    logger.add(log_path, mode="w", encoding="utf-8", format=LOG_FORMAT)
    logger.add(lambda message: print(message, end=""), format="{message}")

    return logger
