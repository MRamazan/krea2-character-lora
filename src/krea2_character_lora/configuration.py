from dataclasses import dataclass, field
from typing import Literal

from .constants import ACCEPTED_IMAGE_EXTENSIONS
from .errors import ConfigurationError

CaptionTriggerPolicy = Literal["require", "warn", "ignore"]
CheckpointMode = Literal["auto", "all", "selected", "manual"]
ResumeMode = Literal["auto", "never", "required"]


@dataclass(slots=True)
class DatasetConfig:
    trigger_word: str
    caption_trigger_policy: CaptionTriggerPolicy = "require"
    auto_prefix_missing_trigger: bool = False
    minimum_pair_count: int = 4
    expected_pair_count: int | None = None
    fail_on_exact_duplicates: bool = True
    near_duplicate_hamming_threshold: int = 8
    accepted_image_extensions: tuple[str, ...] = ACCEPTED_IMAGE_EXTENSIONS
    gallery_columns: int = 4
    gallery_thumbnail_size: int = 320
    gallery_page_size: int = 24

    def validate(self) -> None:
        if not self.trigger_word.strip():
            raise ConfigurationError("The character trigger word must not be empty.")
        if self.caption_trigger_policy not in {"require", "warn", "ignore"}:
            raise ConfigurationError("The caption trigger policy is invalid.")
        if self.minimum_pair_count <= 0:
            raise ConfigurationError("The minimum pair count must be positive.")
        if self.expected_pair_count is not None and self.expected_pair_count <= 0:
            raise ConfigurationError("The expected pair count must be positive when provided.")
        if self.near_duplicate_hamming_threshold < 0:
            raise ConfigurationError("The near-duplicate Hamming threshold must not be negative.")
        if not self.accepted_image_extensions:
            raise ConfigurationError("At least one accepted image extension is required.")
        normalized = {value.lower() for value in self.accepted_image_extensions}
        if any(not value.startswith(".") for value in normalized):
            raise ConfigurationError("Accepted image extensions must start with a period.")
        if self.gallery_columns <= 0 or self.gallery_thumbnail_size <= 0:
            raise ConfigurationError("Gallery layout values must be positive.")
        if self.gallery_page_size <= 0:
            raise ConfigurationError("The gallery page size must be positive.")

    def normalized_extensions(self) -> tuple[str, ...]:
        return tuple(sorted({value.lower() for value in self.accepted_image_extensions}))


@dataclass(slots=True)
class TrainingConfig:
    project_name: str = "krea2_character_lora"
    run_name: str = "character_v1"
    training_steps: int = 2000
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    optimizer: str = "adamw"
    lr_scheduler: str = "constant"
    max_grad_norm: float = 1.0
    batch_size: int = 1
    gradient_accumulation: int = 1
    resolutions: tuple[int, ...] = (768, 1024)
    lora_rank: int = 32
    lora_alpha: int = 32
    save_every: int = 200
    max_checkpoints_to_keep: int = 10
    dataset_repeats: int = 1
    caption_dropout_rate: float = 0.0
    token_dropout_rate: float = 0.0
    shuffle_tokens: bool = False
    keep_tokens: int = 1
    flip_x: bool = False
    training_dtype: str = "bf16"
    generate_training_samples: bool = False
    training_sample_every: int = 200
    raw_sample_steps: int = 52
    raw_sample_guidance: float = 3.5

    def validate(self) -> None:
        if not self.project_name.strip():
            raise ConfigurationError("The project name must not be empty.")
        if not self.run_name.strip():
            raise ConfigurationError("The run name must not be empty.")
        if not _is_safe_name(self.run_name):
            raise ConfigurationError(
                "The run name may contain only letters, numbers, periods, underscores, and hyphens."
            )
        integer_fields = {
            "training_steps": self.training_steps,
            "batch_size": self.batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "save_every": self.save_every,
            "max_checkpoints_to_keep": self.max_checkpoints_to_keep,
            "dataset_repeats": self.dataset_repeats,
            "training_sample_every": self.training_sample_every,
            "raw_sample_steps": self.raw_sample_steps,
        }
        invalid = [name for name, value in integer_fields.items() if value <= 0]
        if invalid:
            raise ConfigurationError(
                f"These fields must be positive: {', '.join(sorted(invalid))}."
            )
        if self.learning_rate <= 0:
            raise ConfigurationError("The learning rate must be positive.")
        if self.weight_decay < 0:
            raise ConfigurationError("The weight decay must not be negative.")
        if self.max_grad_norm <= 0:
            raise ConfigurationError("The maximum gradient norm must be positive.")
        if not self.resolutions or any(value <= 0 for value in self.resolutions):
            raise ConfigurationError("Training resolutions must contain positive values.")
        if self.caption_dropout_rate < 0 or self.caption_dropout_rate > 1:
            raise ConfigurationError("The caption dropout rate must be between zero and one.")
        if self.token_dropout_rate < 0 or self.token_dropout_rate > 1:
            raise ConfigurationError("The token dropout rate must be between zero and one.")


@dataclass(slots=True)
class EvaluationConfig:
    prompts: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [42, 12345, 987654321])
    width: int = 1024
    height: int = 1024
    inference_steps: int = 8
    guidance_scale: float = 0.0
    negative_prompt: str = ""
    checkpoint_mode: CheckpointMode = "auto"
    maximum_checkpoints: int = 8
    manual_checkpoint_steps: list[int] = field(default_factory=list)
    primary_adapter_scale: float = 1.0
    scale_sweep: list[float] = field(default_factory=lambda: [0.6, 0.8, 1.0])
    compare_base_model: bool = True
    include_base_in_checkpoint_grid: bool = True
    run_checkpoint_sweep: bool = True
    run_scale_sweep: bool = True

    def validate(self) -> None:
        if not self.prompts or any(not prompt.strip() for prompt in self.prompts):
            raise ConfigurationError("Evaluation prompts must contain non-empty values.")
        if not self.seeds:
            raise ConfigurationError("At least one evaluation seed is required.")
        if self.width <= 0 or self.height <= 0:
            raise ConfigurationError("Evaluation dimensions must be positive.")
        if self.inference_steps <= 0:
            raise ConfigurationError("Inference steps must be positive.")
        if self.checkpoint_mode not in {"auto", "all", "selected", "manual"}:
            raise ConfigurationError("The checkpoint mode is invalid.")
        if self.checkpoint_mode == "manual" and not self.manual_checkpoint_steps:
            raise ConfigurationError(
                "Manual checkpoint steps are required when the checkpoint mode is manual."
            )
        if self.maximum_checkpoints <= 0:
            raise ConfigurationError("The maximum checkpoint count must be positive.")
        if self.primary_adapter_scale < 0:
            raise ConfigurationError("The primary adapter scale must not be negative.")
        if not self.scale_sweep or any(value < 0 for value in self.scale_sweep):
            raise ConfigurationError("The scale sweep must contain non-negative values.")


def _is_safe_name(value: str) -> bool:
    return all(character.isalnum() or character in {".", "_", "-"} for character in value)
