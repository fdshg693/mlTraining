"""Convert a Jupyter notebook to a Python script.

The generated script is intentionally simple: code cells are kept as Python
code and Markdown cells are written as Python comments. By default the
output is created next to the input notebook with the same stem and a
``.py`` suffix; pass ``--output`` to write it elsewhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _source_text(cell: dict[str, Any]) -> str:
    """Return a cell's source regardless of how nbformat stores it."""

    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    if isinstance(source, str):
        return source
    raise ValueError("cell source must be a string or a list of strings")


def _markdown_to_comments(source: str) -> str:
    """Turn Markdown text into line-oriented Python comments."""

    lines = source.splitlines()
    if not lines:
        return "#"
    return "\n".join(f"# {line}" if line else "#" for line in lines)


def convert_notebook(notebook_path: Path, output_path: Path | None = None) -> Path:
    """Convert *notebook_path* and return the generated Python file path.

    If *output_path* is omitted, the script is written next to the input
    notebook with the same stem and a ``.py`` suffix. If *output_path* ends
    in ``.py`` it is used as the exact output file path; otherwise it is
    treated as a directory and the file is written there using the
    notebook's stem.
    """

    if notebook_path.suffix.lower() != ".ipynb":
        raise ValueError(f"input file must have an .ipynb suffix: {notebook_path}")

    with notebook_path.open(encoding="utf-8") as notebook_file:
        notebook = json.load(notebook_file)

    if not isinstance(notebook, dict):
        raise ValueError("notebook JSON must contain an object at the top level")

    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError("notebook does not contain a valid cells list")

    output_parts: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("notebook contains a cell with an invalid format")

        cell_type = cell.get("cell_type")
        source = _source_text(cell)
        if cell_type == "markdown":
            output_parts.append(_markdown_to_comments(source))
        elif cell_type == "code":
            output_parts.append(source.rstrip("\n"))
        elif cell_type == "raw":
            # Raw cells are not executable Python, so preserve their content
            # safely as comments rather than producing invalid source.
            output_parts.append(_markdown_to_comments(source))
        else:
            raise ValueError(f"unsupported cell type: {cell_type!r}")

    if output_path is None:
        output_path = notebook_path.with_suffix(".py")
    elif output_path.suffix.lower() != ".py":
        output_path = output_path / (notebook_path.stem + ".py")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(output_parts).rstrip() + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Jupyter notebook to a Python script."
    )
    parser.add_argument("notebook", type=Path, help="path to the input .ipynb file")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=(
            "output .py file path, or a directory to place the converted "
            "file in (default: alongside the input notebook)"
        ),
    )
    args = parser.parse_args()

    output_path = convert_notebook(args.notebook, args.output)
    print(output_path)


if __name__ == "__main__":
    main()
