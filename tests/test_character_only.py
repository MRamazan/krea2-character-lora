from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNED = [ROOT / "src", ROOT / "notebooks" / "cells"]


def test_character_pipeline_has_no_concept_type() -> None:
    violations = []
    for root in SCANNED:
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8").lower()
            if "concept_type" in source:
                violations.append(str(path.relative_to(ROOT)))
    assert not violations
