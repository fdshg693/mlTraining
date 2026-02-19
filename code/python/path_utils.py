from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"


def input_path(*parts: str) -> Path:
    """data/input 配下のパスを返す"""
    return DATA_DIR / "input" / Path(*parts)


def output_path(*parts: str) -> Path:
    """data/output 配下のパスを返す"""
    return DATA_DIR / "output" / Path(*parts)
