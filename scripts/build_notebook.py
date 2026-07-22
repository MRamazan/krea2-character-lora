import argparse
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
CELL_DIRECTORY = ROOT / "notebooks" / "cells"
OUTPUT_PATH = ROOT / "notebooks" / "krea2_character_lora_colab.ipynb"


def build_notebook() -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata = {
        "colab": {"name": "krea2_character_lora_colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    notebook.cells = [
        nbformat.v4.new_code_cell(path.read_text(encoding="utf-8").rstrip())
        for path in sorted(CELL_DIRECTORY.glob("*.py"))
    ]
    return notebook


def normalized(notebook: nbformat.NotebookNode) -> dict:
    payload = json.loads(nbformat.writes(notebook))
    for cell in payload.get("cells", []):
        cell["execution_count"] = None
        cell["outputs"] = []
        cell["id"] = "stable"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    generated = build_notebook()
    if arguments.check:
        if not OUTPUT_PATH.is_file():
            raise SystemExit("The generated notebook is missing.")
        existing = nbformat.read(OUTPUT_PATH, as_version=4)
        if normalized(existing) != normalized(generated):
            raise SystemExit("The generated notebook is out of date.")
        return
    nbformat.write(generated, OUTPUT_PATH)


if __name__ == "__main__":
    main()
