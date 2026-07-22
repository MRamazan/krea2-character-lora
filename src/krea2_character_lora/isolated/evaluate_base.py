import json
import os
from pathlib import Path

import numpy as np
from krea2_runtime import (
    capture_base_parameter_samples,
    compare_base_parameter_samples,
    load_runtime,
    sha256_file,
    unload_runtime,
)
from PIL import Image, ImageDraw
from toolkit.config_modules import GenerateImageConfig

request = json.loads(Path(os.environ["KREA2_EVAL_REQUEST"]).read_text(encoding="utf-8"))
assets = json.loads(Path(request["inference_asset_manifest"]).read_text(encoding="utf-8"))
selection = request["active_checkpoint"]
output_directory = Path(request["output_root"]) / "base_comparison"
output_directory.mkdir(parents=True, exist_ok=True)

adapter_specs = [
    {
        "name": request["adapter_name"],
        "path": selection["checkpoint_path"],
        "scale": request["primary_adapter_scale"],
        "alpha": request["lora_alpha"],
    }
]
model, transformer, controller, records = load_runtime(
    assets,
    adapter_specs,
    dtype=request["training_dtype"],
    max_text_length=request["max_text_length"],
)
base_samples = capture_base_parameter_samples(transformer)
base_scales = {request["adapter_name"]: 0.0}
active_scales = {request["adapter_name"]: request["primary_adapter_scale"]}
prompt = request["prompts"][0]
seed = int(request["seeds"][0])
base_path = output_directory / "base.png"
active_path = output_directory / "active_lora.png"


def generate(path, scales):
    controller.set_scales(scales)
    model.generate_images(
        [
            GenerateImageConfig(
                prompt=prompt,
                width=int(request["width"]),
                height=int(request["height"]),
                num_inference_steps=int(request["inference_steps"]),
                guidance_scale=float(request["guidance_scale"]),
                negative_prompt=request["negative_prompt"],
                seed=seed,
                network_multiplier=1.0,
                output_path=str(path),
                output_ext="png",
                add_prompt_file=False,
            )
        ]
    )


generate(base_path, base_scales)
generate(active_path, active_scales)
if not base_path.is_file() or not active_path.is_file():
    raise RuntimeError("The base comparison images were not created.")
base_image = Image.open(base_path).convert("RGB")
active_image = Image.open(active_path).convert("RGB")
pixel_mae = float(
    np.mean(np.abs(np.asarray(base_image, np.float32) - np.asarray(active_image, np.float32)))
)
if pixel_mae <= 0.0:
    raise RuntimeError("The base and active-adapter outputs are identical.")
label_height = 52
grid = Image.new(
    "RGB",
    (
        base_image.width + active_image.width,
        max(base_image.height, active_image.height) + label_height,
    ),
    "white",
)
grid.paste(base_image, (0, label_height))
grid.paste(active_image, (base_image.width, label_height))
draw = ImageDraw.Draw(grid)
draw.text((18, 17), "Turbo base", fill="black")
draw.text((base_image.width + 18, 17), "Turbo with character LoRA", fill="black")
grid_path = output_directory / "base_vs_lora_grid.png"
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
    "pixel_mae": pixel_mae,
    "base_path": str(base_path),
    "active_path": str(active_path),
    "grid_path": str(grid_path),
    "grid_sha256": sha256_file(grid_path),
    "base_parameters_unchanged": base_unchanged,
    "base_parameter_records": base_parameter_records,
    "permanent_merge_performed": False,
}
metadata_path = output_directory / "base_comparison_metadata.json"
metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print("__KREA2_RESULT__" + json.dumps(metadata))
unload_runtime(model, transformer, controller, records)
