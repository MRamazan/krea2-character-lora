import json
import os
import subprocess
from pathlib import Path

preflight_script = r"""
import json
import os
from pathlib import Path
import yaml
from toolkit.config_modules import ModelConfig, NetworkConfig
from extensions_built_in.diffusion_models.krea2.krea2 import Krea2Model

configuration_path = Path(os.environ["KREA2_TRAINING_CONFIGURATION"])
configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
processes = configuration.get("config", {}).get("process", [])
if not isinstance(processes, list) or len(processes) != 1:
    raise RuntimeError("The configuration must contain exactly one process.")
process = processes[0]
if process.get("type") != "sd_trainer":
    raise RuntimeError("The process type must be sd_trainer.")
model_section = process.get("model", {})
network_section = process.get("network", {})
train_section = process.get("train", {})
dataset_section = process.get("datasets", [])
if model_section.get("arch") != "krea2":
    raise RuntimeError("The model architecture must be krea2.")
if network_section.get("type") != "lora":
    raise RuntimeError("This notebook supports LoRA network training only.")
if not network_section.get("transformer_only"):
    raise RuntimeError("Krea 2 LoRA training must target the transformer.")
if train_section.get("train_text_encoder"):
    raise RuntimeError("Krea 2 text-encoder training is not supported by this pipeline.")
if train_section.get("merge_network_on_save"):
    raise RuntimeError("Permanent LoRA merging must remain disabled.")
if not isinstance(dataset_section, list) or len(dataset_section) != 1:
    raise RuntimeError("Exactly one canonical dataset directory is required.")
dataset_directory = Path(dataset_section[0]["folder_path"])
images = [path for path in dataset_directory.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
captions = list(dataset_directory.glob("*.txt"))
if not images or len(images) != len(captions):
    raise RuntimeError("The canonical dataset directory does not contain matching image-caption pairs.")
model_configuration = ModelConfig(**model_section)
network_configuration = NetworkConfig(**network_section)
probe = Krea2Model(device="cuda:0", model_config=model_configuration, dtype=train_section.get("dtype", "bf16"))
result = {
    "run_name": configuration["config"]["name"],
    "dataset_pairs": len(images),
    "model_arch": probe.arch,
    "target_lora_modules": list(probe.target_lora_modules),
    "network_type": network_configuration.type,
    "network_rank": network_configuration.linear,
    "network_alpha": network_configuration.linear_alpha,
    "training_steps": train_section.get("steps"),
    "train_text_encoder": train_section.get("train_text_encoder"),
    "merge_network_on_save": train_section.get("merge_network_on_save"),
    "checkpoint_filename": model_section.get("model_kwargs", {}).get("checkpoint_filename"),
    "text_encoder_path": model_section.get("model_kwargs", {}).get("text_encoder_path"),
    "vae_path": model_section.get("model_kwargs", {}).get("vae_path"),
}
print(json.dumps(result))
"""

environment = os.environ.copy()
environment["KREA2_TRAINING_CONFIGURATION"] = str(PATHS["config"] / "train_krea2_lora.yaml")
result = subprocess.run(
    [str(PATHS["venv_python"]), "-c", preflight_script],
    cwd=str(PATHS["ai_toolkit"]),
    env=environment,
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    raise RuntimeError(f"Training preflight failed.\n{result.stdout}\n{result.stderr}")
preflight = json.loads(result.stdout.strip().splitlines()[-1])
if preflight["target_lora_modules"] != ["SingleStreamDiT"]:
    raise RuntimeError(f"Unexpected Krea 2 LoRA target modules: {preflight['target_lora_modules']}")
preflight_path = PATHS["config"] / "training_preflight.json"
preflight_path.write_text(json.dumps(preflight, indent=2), encoding="utf-8")
print(json.dumps(preflight, indent=2))
print(f"Training preflight: {preflight_path}")