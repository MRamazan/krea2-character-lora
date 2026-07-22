import pytest

from krea2_character_lora import DatasetConfig, EvaluationConfig, TrainingConfig
from krea2_character_lora.errors import ConfigurationError


def test_dataset_config_requires_trigger_word() -> None:
    with pytest.raises(ConfigurationError):
        DatasetConfig(trigger_word="").validate()


def test_training_config_accepts_character_defaults() -> None:
    TrainingConfig().validate()


def test_evaluation_config_requires_prompts() -> None:
    with pytest.raises(ConfigurationError):
        EvaluationConfig().validate()
