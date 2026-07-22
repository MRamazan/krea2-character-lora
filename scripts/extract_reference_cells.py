import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reference" / "universal_pipeline_v2_1.ipynb"
DESTINATION = ROOT / "reference" / "extracted_cells"


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    headings = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        source = "".join(cell.get("source", []))
        match = re.search(r"^## Cell (\d+) — (.+)$", source, flags=re.MULTILINE)
        if match:
            headings[index + 1] = (int(match.group(1)), match.group(2).strip())
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for path in DESTINATION.glob("*.py"):
        path.unlink()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        number, title = headings.get(index, (index, f"Code cell {index}"))
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        path = DESTINATION / f"{number:02d}_{slug}.py"
        path.write_text("".join(cell.get("source", [])), encoding="utf-8")


if __name__ == "__main__":
    main()
