import json
import re
from pathlib import Path

PROJECT_ROOT = Path("/content/krea2_lora").resolve()

USER_CONFIG = {
    "project_name": "my_krea2_lora_project",
    "run_name": "my_krea2_lora_v1",
    "concept_type": "character",
    "trigger_word": "myconcept",
    "expected_pair_count": None,
    "minimum_pair_count": 4,
    "caption_trigger_policy": "require",
    "auto_prefix_missing_trigger": False,
    "fail_on_exact_duplicates": True,
    "near_duplicate_hamming_threshold": 8,
    "ai_toolkit_repository": "https://github.com/ostris/ai-toolkit.git",
    "ai_toolkit_revision": "main",
    "force_reinstall_environment": False,
    "virtualenv_version": "21.6.1",
    "torch_version": "2.9.1",
    "torchvision_version": "0.24.1",
    "torchaudio_version": "2.9.1",
    "torch_index_url": "https://download.pytorch.org/whl/cu128",
    "training_model_repository": "krea/Krea-2-Raw",
    "training_model_revision": None,
    "training_checkpoint_filename": "raw.safetensors",
    "inference_model_repository": "krea/Krea-2-Turbo",
    "inference_model_revision": None,
    "inference_checkpoint_filename": "turbo.safetensors",
    "text_encoder_repository": "Qwen/Qwen3-VL-4B-Instruct",
    "text_encoder_revision": None,
    "vae_repository": "Qwen/Qwen-Image",
    "vae_revision": None,
    "max_text_length": 512,
    "training_resolutions": [768, 1024],
    "batch_size": 1,
    "gradient_accumulation": 1,
    "training_steps": 2000,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "optimizer": "adamw",
    "lr_scheduler": "constant",
    "max_grad_norm": 1.0,
    "lora_rank": 32,
    "lora_alpha": 32,
    "save_every": 200,
    "max_step_saves_to_keep": 50,
    "dataset_repeats": 1,
    "caption_dropout_rate": 0.0,
    "token_dropout_rate": 0.0,
    "shuffle_tokens": False,
    "keep_tokens": 1,
    "flip_x": False,
    "cache_latents_to_disk": True,
    "cache_text_embeddings": True,
    "training_dtype": "bf16",
    "quantize_transformer": False,
    "quantize_text_encoder": False,
    "low_vram": False,
    "layer_offloading": False,
    "run_training_smoke_test": True,
    "smoke_test_steps": 3,
    "run_production_training": True,
    "disable_training_samples": True,
    "training_sample_every": 200,
    "raw_sample_steps": 52,
    "raw_sample_guidance": 3.5,
    "evaluation_prompts": [
        "myconcept in a neutral studio photograph with clear subject visibility",
        "myconcept in a natural outdoor environment with realistic lighting",
        "myconcept in a wide composition with a different pose or presentation",
    ],
    "evaluation_seeds": [42, 12345, 987654321],
    "inference_width": 1024,
    "inference_height": 1024,
    "inference_steps": 8,
    "inference_guidance": 0.0,
    "negative_prompt": "",
    "primary_adapter_name": "concept_adapter",
    "primary_adapter_scale": 1.0,
    "additional_loras": [],
    "checkpoint_selection_mode": "final_if_present_else_latest",
    "manual_checkpoint_step": None,
    "checkpoint_sweep_mode": "auto",
    "manual_sweep_steps": [],
    "maximum_sweep_checkpoints": 8,
    "scale_sweep": [0.6, 0.8, 1.0],
    "run_inference": True,
    "run_checkpoint_sweep": True,
    "run_scale_sweep": True,
    "checkpoints_per_archive": 4,
    "auto_download_exports": False,
    "strict_hardware_check": False,
    "minimum_gpu_memory_gib": 40,
}

if not re.fullmatch(r"[A-Za-z0-9._-]+", USER_CONFIG["run_name"]):
    raise RuntimeError("run_name may contain only letters, numbers, periods, underscores, and hyphens.")

if USER_CONFIG["caption_trigger_policy"] not in {"require", "warn", "ignore"}:
    raise RuntimeError("caption_trigger_policy must be require, warn, or ignore.")

if USER_CONFIG["checkpoint_selection_mode"] not in {"final_if_present_else_latest", "latest", "manual"}:
    raise RuntimeError("checkpoint_selection_mode is invalid.")

if USER_CONFIG["checkpoint_sweep_mode"] not in {"auto", "all", "manual", "selected_only"}:
    raise RuntimeError("checkpoint_sweep_mode is invalid.")

positive_integer_fields = [
    "minimum_pair_count",
    "max_text_length",
    "batch_size",
    "gradient_accumulation",
    "training_steps",
    "lora_rank",
    "lora_alpha",
    "save_every",
    "max_step_saves_to_keep",
    "dataset_repeats",
    "smoke_test_steps",
    "inference_width",
    "inference_height",
    "inference_steps",
    "maximum_sweep_checkpoints",
    "checkpoints_per_archive",
]
for field in positive_integer_fields:
    if not isinstance(USER_CONFIG[field], int) or USER_CONFIG[field] <= 0:
        raise RuntimeError(f"{field} must be a positive integer.")

if USER_CONFIG["expected_pair_count"] is not None:
    if not isinstance(USER_CONFIG["expected_pair_count"], int) or USER_CONFIG["expected_pair_count"] <= 0:
        raise RuntimeError("expected_pair_count must be null or a positive integer.")

if not USER_CONFIG["trigger_word"].strip() and USER_CONFIG["caption_trigger_policy"] == "require":
    raise RuntimeError("A non-empty trigger_word is required when caption_trigger_policy is require.")

if not USER_CONFIG["evaluation_prompts"] or not all(isinstance(prompt, str) and prompt.strip() for prompt in USER_CONFIG["evaluation_prompts"]):
    raise RuntimeError("evaluation_prompts must contain at least one non-empty prompt.")

if not USER_CONFIG["evaluation_seeds"] or not all(isinstance(seed, int) for seed in USER_CONFIG["evaluation_seeds"]):
    raise RuntimeError("evaluation_seeds must contain at least one integer seed.")

if not USER_CONFIG["training_resolutions"] or not all(isinstance(value, int) and value > 0 for value in USER_CONFIG["training_resolutions"]):
    raise RuntimeError("training_resolutions must contain positive integers.")

if not USER_CONFIG["scale_sweep"] or not all(isinstance(value, (int, float)) and value >= 0 for value in USER_CONFIG["scale_sweep"]):
    raise RuntimeError("scale_sweep must contain one or more non-negative values.")

if not isinstance(USER_CONFIG["primary_adapter_scale"], (int, float)) or USER_CONFIG["primary_adapter_scale"] < 0:
    raise RuntimeError("primary_adapter_scale must be non-negative.")

if USER_CONFIG["trigger_word"]:
    USER_CONFIG["evaluation_prompts"] = [
        prompt.replace("myconcept", USER_CONFIG["trigger_word"])
        for prompt in USER_CONFIG["evaluation_prompts"]
    ]

adapter_names = {USER_CONFIG["primary_adapter_name"]}
if not USER_CONFIG["primary_adapter_name"].strip():
    raise RuntimeError("primary_adapter_name must not be empty.")
for item in USER_CONFIG["additional_loras"]:
    if not isinstance(item, dict):
        raise RuntimeError("Every additional_loras entry must be a dictionary.")
    missing = {"name", "path", "scale"} - set(item)
    if missing:
        raise RuntimeError(f"An additional LoRA entry is missing fields: {sorted(missing)}")
    if item["name"] in adapter_names:
        raise RuntimeError(f"Duplicate adapter name: {item['name']}")
    if float(item["scale"]) < 0:
        raise RuntimeError(f"Additional LoRA scale must be non-negative: {item['name']}")
    adapter_names.add(item["name"])

PATHS = {
    "root": PROJECT_ROOT,
    "ai_toolkit": PROJECT_ROOT / "ai-toolkit",
    "venv": PROJECT_ROOT / "venv",
    "venv_python": PROJECT_ROOT / "venv" / "bin" / "python",
    "config": PROJECT_ROOT / "config",
    "logs": PROJECT_ROOT / "logs",
    "models": PROJECT_ROOT / "models",
    "dataset": PROJECT_ROOT / "dataset",
    "dataset_raw": PROJECT_ROOT / "dataset" / "raw",
    "dataset_training": PROJECT_ROOT / "dataset" / "training",
    "dataset_audit": PROJECT_ROOT / "dataset" / "audit",
    "checkpoints": PROJECT_ROOT / "checkpoints",
    "smoke_checkpoints": PROJECT_ROOT / "smoke_checkpoints",
    "inference": PROJECT_ROOT / "inference",
    "exports": PROJECT_ROOT / "exports",
    "helpers": PROJECT_ROOT / "runtime_helpers",
}

for key in [
    "root",
    "config",
    "logs",
    "models",
    "dataset",
    "dataset_audit",
    "checkpoints",
    "smoke_checkpoints",
    "inference",
    "exports",
    "helpers",
]:
    PATHS[key].mkdir(parents=True, exist_ok=True)

configuration_path = PATHS["config"] / "user_configuration.json"
configuration_path.write_text(json.dumps(USER_CONFIG, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps(USER_CONFIG, indent=2, ensure_ascii=False))
print(f"Configuration saved to: {configuration_path}")