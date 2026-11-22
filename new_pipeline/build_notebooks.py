#!/usr/bin/env python3
"""Build clean Jupyter notebooks from the cell-marked .py sources.

The .py file is the source of truth. A cell starts with:

    # %%                 -> code cell
    # %% [markdown]      -> markdown cell (following lines may be "# text")

Run:
    python new_pipeline/build_notebooks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
NOTEBOOKS = ROOT / "notebooks"


def parse_cells(text: str) -> list[dict]:
    cells: list[dict] = []
    current_kind: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_kind, current_lines
        if current_kind is None:
            return
        lines = current_lines
        if current_kind == "markdown":
            cleaned = []
            for line in lines:
                if line.startswith("# "):
                    cleaned.append(line[2:])
                elif line == "#":
                    cleaned.append("")
                else:
                    cleaned.append(line)
            source = "\n".join(cleaned).strip("\n")
            if source:
                cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            source = "\n".join(lines).strip("\n")
            if source:
                cells.append(
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": source,
                    }
                )
        current_kind = None
        current_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush()
            current_kind = "markdown" if "[markdown]" in stripped else "code"
            continue
        if current_kind is not None:
            current_lines.append(line)
    flush()
    return cells


def build(source: Path, output: Path) -> None:
    text = source.read_text(encoding="utf-8")
    cells = parse_cells(text)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"{source.name} -> {output.relative_to(ROOT)} ({len(cells)} cells)")


def main(argv: list[str]) -> int:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    sources = sorted(SRC.glob("*.py")) if not argv else [Path(arg) for arg in argv]
    for source in sources:
        if not source.is_file():
            print(f"missing: {source}", file=sys.stderr)
            return 1
        build(source, NOTEBOOKS / f"{source.stem}.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
