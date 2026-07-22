import inspect

from krea2_character_lora import CharacterLoraPipeline, ImportedRun, TrainingRun


def test_pipeline_has_required_public_operations() -> None:
    required = {
        "setup",
        "prepare_dataset",
        "load_dataset",
        "preview_training",
        "train",
        "load_run",
        "prepare_evaluation_assets",
        "evaluate",
        "import_evaluation_bundle",
    }
    assert required <= set(dir(CharacterLoraPipeline))


def test_import_evaluation_bundle_takes_single_zip_path() -> None:
    signature = inspect.signature(CharacterLoraPipeline.import_evaluation_bundle)
    parameters = [name for name in signature.parameters if name != "self"]
    assert parameters == ["zip_path"]


def test_imported_run_is_a_training_run() -> None:
    assert issubclass(ImportedRun, TrainingRun)


def test_pipeline_constructor_has_no_concept_selector() -> None:
    signature = inspect.signature(CharacterLoraPipeline)
    assert "concept_type" not in signature.parameters


def test_pipeline_setup_has_no_token_parameter() -> None:
    signature = inspect.signature(CharacterLoraPipeline.setup)
    parameters = set(signature.parameters)
    assert "hf_token" not in parameters
    assert not any("token" in name.lower() for name in parameters)


def test_evaluation_asset_preparation_takes_no_token() -> None:
    signature = inspect.signature(CharacterLoraPipeline.prepare_evaluation_assets)
    assert not any("token" in name.lower() for name in signature.parameters)
