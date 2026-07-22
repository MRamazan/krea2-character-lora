import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "krea2_character_lora_colab.ipynb"
EVALUATION_NOTEBOOK = ROOT / "notebooks" / "krea2_character_lora_evaluation_colab.ipynb"

TOKEN_MARKERS = (
    "hf_token",
    "HF_TOKEN",
    "getpass",
    "userdata.get",
    "colab import userdata",
    "whoami",
    ".env",
)


def code_cells(notebook_path: Path = NOTEBOOK) -> list[str]:
    notebook = nbformat.read(notebook_path, as_version=4)
    return [cell.source for cell in notebook.cells if cell.cell_type == "code"]


def evaluation_cells() -> list[str]:
    return code_cells(EVALUATION_NOTEBOOK)


def test_notebook_has_exactly_four_code_cells() -> None:
    assert len(code_cells()) == 4


def test_setup_cell_installs_package_and_initializes_pipeline() -> None:
    setup = code_cells()[0]
    assert "REPOSITORY_URL" in setup
    assert "PIPELINE_REVISION" in setup
    assert "CharacterLoraPipeline" in setup
    assert "pipeline.setup(" in setup


def test_notebook_has_no_hugging_face_authentication() -> None:
    source = "\n".join(code_cells())
    for marker in TOKEN_MARKERS:
        assert marker not in source


def test_dataset_cell_owns_trigger_word_and_gallery() -> None:
    dataset = code_cells()[1]
    assert re.search(r'^TRIGGER_WORD\s*=\s*".+"', dataset, flags=re.MULTILINE)
    assert "show_gallery" in dataset
    assert "show_caption_audit" in dataset
    assert "show_issues" in dataset


def test_training_and_evaluation_reuse_persisted_trigger() -> None:
    training = code_cells()[2]
    evaluation = code_cells()[3]
    assert "TRIGGER_WORD" not in training
    assert "TRIGGER_WORD" not in evaluation
    assert "training_run.trigger_word" in evaluation


def test_evaluation_cell_exports() -> None:
    evaluation = code_cells()[3]
    assert "prepare_evaluation_assets" in evaluation
    assert "evaluation.export(" in evaluation


def test_notebook_has_no_drive_integration() -> None:
    source = "\n".join(code_cells()).lower()
    assert "google.colab import drive" not in source
    assert "drive.mount" not in source


def test_notebook_has_no_concept_selector() -> None:
    source = "\n".join(code_cells()).lower()
    assert "concept_type" not in source


def test_evaluation_notebook_has_exactly_three_code_cells() -> None:
    assert len(evaluation_cells()) == 3


def test_evaluation_notebook_setup_import_evaluation_stages() -> None:
    cells = evaluation_cells()
    setup, importer, evaluation = cells
    assert "CharacterLoraPipeline" in setup
    assert "prepare_training_assets=False" in setup
    assert "import_evaluation_bundle" in importer
    assert "from google.colab import files" in importer
    assert "imported_run.display_summary()" in importer
    assert "prepare_evaluation_assets" in evaluation
    assert "evaluation.export(" in evaluation


def test_evaluation_notebook_derives_trigger_from_imported_run() -> None:
    evaluation = evaluation_cells()[2]
    assert "TRIGGER_WORD = imported_run.trigger_word" in evaluation
    assert 'f"{TRIGGER_WORD} is a woman' in evaluation


def test_evaluation_notebook_has_no_hugging_face_authentication() -> None:
    source = "\n".join(evaluation_cells())
    for marker in TOKEN_MARKERS:
        assert marker not in source


def test_evaluation_notebook_has_no_drive_integration() -> None:
    source = "\n".join(evaluation_cells()).lower()
    assert "google.colab import drive" not in source
    assert "drive.mount" not in source
