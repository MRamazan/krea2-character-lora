import argparse
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"

NOTEBOOKS = [
    {
        "name": "krea2_character_lora_colab.ipynb",
        "cells": NOTEBOOKS_DIR / "cells",
    },
    {
        "name": "krea2_character_lora_evaluation_colab.ipynb",
        "cells": NOTEBOOKS_DIR / "evaluation_cells",
    },
]


def build_notebook(cell_directory: Path, name: str) -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata = {
        "colab": {"name": name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    notebook.cells = [
        nbformat.v4.new_code_cell(path.read_text(encoding="utf-8").rstrip())
        for path in sorted(cell_directory.glob("*.py"))
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
    for specification in NOTEBOOKS:
        output_path = NOTEBOOKS_DIR / specification["name"]
        generated = build_notebook(specification["cells"], specification["name"])
        if arguments.check:
            if not output_path.is_file():
                raise SystemExit(f"The generated notebook is missing: {specification['name']}")
            existing = nbformat.read(output_path, as_version=4)
            if normalized(existing) != normalized(generated):
                raise SystemExit(f"The generated notebook is out of date: {specification['name']}")
        else:
            nbformat.write(generated, output_path)


if __name__ == "__main__":
    main()
