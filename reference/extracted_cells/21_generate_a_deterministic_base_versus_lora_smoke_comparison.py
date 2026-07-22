import json
import os
import subprocess
from pathlib import Path
from IPython.display import display
from PIL import Image

if not USER_CONFIG["run_inference"]:
    print("Inference smoke comparison skipped by configuration.")
else:
    output_directory = PATHS["inference"] / "smoke_comparison"
    output_directory.mkdir(parents=True, exist_ok=True)
    script = r"""
import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from toolkit.config_modules import GenerateImageConfig
from runtime_helpers.krea2_runtime import capture_base_parameter_samples, compare_base_parameter_samples, load_runtime, sha256_file, unload_runtime

root = Path(os.environ["KREA2_PROJECT_ROOT"])
config = json.loads((root / "config" / "user_configuration.json").read_text(encoding="utf-8"))
selection = json.loads((root / "config" / "active_checkpoint_selection.json").read_text(encoding="utf-8"))
assets = json.loads((root / "config" / "inference_asset_manifest.json").read_text(encoding="utf-8"))
output_directory = Path(os.environ["KREA2_OUTPUT_DIRECTORY"])
additional = config["additional_loras"]
adapter_specs = [{
    "name": config["primary_adapter_name"],
    "path": selection["checkpoint_path"],
    "scale": config["primary_adapter_scale"],
    "alpha": config["lora_alpha"],
}] + additional
model, transformer, controller, records = load_runtime(assets, adapter_specs, dtype=config["training_dtype"], max_text_length=config["max_text_length"])
base_samples = capture_base_parameter_samples(transformer)
base_scales = {record["name"]: 0.0 for record in records}
active_scales = {config["primary_adapter_name"]: config["primary_adapter_scale"]}
for item in additional:
    active_scales[item["name"]] = float(item.get("scale", 1.0))
prompt = config["evaluation_prompts"][0]
seed = int(config["evaluation_seeds"][0])
base_path = output_directory / "base.png"
active_path = output_directory / "active_loras.png"
controller.set_scales(base_scales)
model.generate_images([GenerateImageConfig(
    prompt=prompt,
    width=int(config["inference_width"]),
    height=int(config["inference_height"]),
    num_inference_steps=int(config["inference_steps"]),
    guidance_scale=float(config["inference_guidance"]),
    negative_prompt=config["negative_prompt"],
    seed=seed,
    network_multiplier=1.0,
    output_path=str(base_path),
    output_ext="png",
    add_prompt_file=False,
)])
controller.set_scales(active_scales)
model.generate_images([GenerateImageConfig(
    prompt=prompt,
    width=int(config["inference_width"]),
    height=int(config["inference_height"]),
    num_inference_steps=int(config["inference_steps"]),
    guidance_scale=float(config["inference_guidance"]),
    negative_prompt=config["negative_prompt"],
    seed=seed,
    network_multiplier=1.0,
    output_path=str(active_path),
    output_ext="png",
    add_prompt_file=False,
)])
if not base_path.is_file() or not active_path.is_file():
    raise RuntimeError("The smoke comparison images were not created.")
base_image = Image.open(base_path).convert("RGB")
active_image = Image.open(active_path).convert("RGB")
base_array = np.asarray(base_image, dtype=np.float32)
active_array = np.asarray(active_image, dtype=np.float32)
pixel_mae = float(np.mean(np.abs(base_array - active_array)))
if pixel_mae <= 0.0:
    raise RuntimeError("The base and active-adapter outputs are identical.")
label_height = 52
grid = Image.new("RGB", (base_image.width + active_image.width, max(base_image.height, active_image.height) + label_height), "white")
grid.paste(base_image, (0, label_height))
grid.paste(active_image, (base_image.width, label_height))
draw = ImageDraw.Draw(grid)
draw.text((18, 17), "Turbo base", fill="black")
draw.text((base_image.width + 18, 17), "Turbo with active LoRAs", fill="black")
grid_path = output_directory / "base_vs_active_grid.png"
grid.save(grid_path)
base_unchanged, base_parameter_records = compare_base_parameter_samples(transformer, base_samples)
if not base_unchanged:
    raise RuntimeError("Representative Turbo base parameters changed during inference.")
if any(record["network"].can_merge_in or record["network"].is_merged_in for record in records):
    raise RuntimeError("At least one adapter entered a merge-capable or merged state.")
metadata = {
    "checkpoint_step": selection["checkpoint_step"],
    "checkpoint_path": selection["checkpoint_path"],
    "source_kind": selection["source_kind"],
    "training_complete": selection["training_complete"],
    "final_quality_claim_allowed": selection["final_quality_claim_allowed"],
    "prompt": prompt,
    "seed": seed,
    "width": config["inference_width"],
    "height": config["inference_height"],
    "num_inference_steps": config["inference_steps"],
    "guidance_scale": config["inference_guidance"],
    "base_scales": base_scales,
    "active_scales": active_scales,
    "pixel_mae": pixel_mae,
    "base_path": str(base_path),
    "active_path": str(active_path),
    "grid_path": str(grid_path),
    "grid_sha256": sha256_file(grid_path),
    "base_parameters_unchanged": base_unchanged,
    "base_parameter_records": base_parameter_records,
    "permanent_merge_performed": False,
}
metadata_path = output_directory / "smoke_comparison_metadata.json"
metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print("__KREA2_RESULT__" + json.dumps(metadata))
unload_runtime(model, transformer, controller, records)
"""
    environment = os.environ.copy()
    environment["KREA2_PROJECT_ROOT"] = str(PROJECT_ROOT)
    environment["KREA2_OUTPUT_DIRECTORY"] = str(output_directory)
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    environment["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [str(PATHS["venv_python"]), "-c", script],
        cwd=str(PATHS["ai_toolkit"]),
        env=environment,
        capture_output=True,
        text=True,
    )
    log_path = PATHS["logs"] / "inference_smoke_comparison.log"
    log_path.write_text(result.stdout + result.stderr + f"\nExit code: {result.returncode}\n", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Inference smoke comparison failed.\n{result.stdout}\n{result.stderr}\nComplete log: {log_path}")
    structured = [line for line in result.stdout.splitlines() if line.startswith("__KREA2_RESULT__")]
    if len(structured) != 1:
        raise RuntimeError("The inference smoke comparison returned an invalid structured result.")
    metadata = json.loads(structured[0].removeprefix("__KREA2_RESULT__"))
    print(json.dumps(metadata, indent=2))
    display(Image.open(metadata["grid_path"]).convert("RGB"))