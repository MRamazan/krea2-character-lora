import gc
import json
import os
from pathlib import Path

from krea2_runtime import load_runtime, sha256_file, unload_runtime
from PIL import Image, ImageDraw
from safetensors.torch import load_file
from toolkit.config_modules import GenerateImageConfig

request = json.loads(Path(os.environ["KREA2_EVAL_REQUEST"]).read_text(encoding="utf-8"))
assets = json.loads(Path(request["inference_asset_manifest"]).read_text(encoding="utf-8"))
checkpoints = request["sweep_checkpoints"]
if not checkpoints:
    raise RuntimeError("The checkpoint sweep received no checkpoints.")
output_directory = Path(request["output_root"]) / "checkpoint_sweep"
output_directory.mkdir(parents=True, exist_ok=True)

adapter_specs = [
    {
        "name": request["adapter_name"],
        "path": checkpoints[0]["path"],
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
primary_record = records[0]
active_scales = {request["adapter_name"]: request["primary_adapter_scale"]}
controller.set_scales(active_scales)
image_records = []
paths_by_case = {index: [] for index in range(len(request["prompts"]))}
for checkpoint in checkpoints:
    state_dict = load_file(checkpoint["path"])
    state_dict = model.convert_lora_weights_before_load(state_dict)
    primary_record["network"].load_weights(state_dict)
    controller.set_scales(active_scales)
    configurations = []
    expected = []
    for case_index, prompt in enumerate(request["prompts"]):
        seed = int(request["seeds"][case_index % len(request["seeds"])])
        output_path = (
            output_directory / f"case_{case_index + 1:02d}_step_{int(checkpoint['step']):08d}.png"
        )
        configurations.append(
            GenerateImageConfig(
                prompt=prompt,
                width=int(request["width"]),
                height=int(request["height"]),
                num_inference_steps=int(request["inference_steps"]),
                guidance_scale=float(request["guidance_scale"]),
                negative_prompt=request["negative_prompt"],
                seed=seed,
                network_multiplier=1.0,
                output_path=str(output_path),
                output_ext="png",
                add_prompt_file=False,
            )
        )
        expected.append((case_index, prompt, seed, output_path))
    model.generate_images(configurations)
    for case_index, prompt, seed, output_path in expected:
        if not output_path.is_file():
            raise RuntimeError(f"Expected sweep image is missing: {output_path}")
        image_records.append(
            {
                "case_index": case_index + 1,
                "checkpoint_step": int(checkpoint["step"]),
                "checkpoint_path": checkpoint["path"],
                "prompt": prompt,
                "seed": seed,
                "path": str(output_path),
                "sha256": sha256_file(output_path),
            }
        )
        paths_by_case[case_index].append(
            {"step": int(checkpoint["step"]), "path": str(output_path)}
        )
    gc.collect()
thumbnail = 384
label_height = 48
grid_paths = []
master_rows = []
for case_index in range(len(request["prompts"])):
    items = sorted(paths_by_case[case_index], key=lambda item: item["step"])
    grid = Image.new("RGB", (thumbnail * len(items), thumbnail + label_height), "white")
    draw = ImageDraw.Draw(grid)
    for column, item in enumerate(items):
        image = (
            Image.open(item["path"])
            .convert("RGB")
            .resize((thumbnail, thumbnail), Image.Resampling.LANCZOS)
        )
        grid.paste(image, (column * thumbnail, label_height))
        draw.text((column * thumbnail + 12, 16), f"Step {item['step']}", fill="black")
    grid_path = output_directory / f"case_{case_index + 1:02d}_checkpoint_grid.png"
    grid.save(grid_path)
    grid_paths.append(str(grid_path))
    master_rows.append(grid)
master_width = max(row.width for row in master_rows)
master_height = sum(row.height for row in master_rows)
master = Image.new("RGB", (master_width, master_height), "white")
offset = 0
for row in master_rows:
    master.paste(row, (0, offset))
    offset += row.height
master_path = output_directory / "checkpoint_master_grid.png"
master.save(master_path)
metadata = {
    "checkpoint_steps": [int(checkpoint["step"]) for checkpoint in checkpoints],
    "prompt_count": len(request["prompts"]),
    "image_count": len(image_records),
    "character_scale": request["primary_adapter_scale"],
    "records": image_records,
    "per_prompt_grids": grid_paths,
    "master_grid": str(master_path),
    "automatic_best_checkpoint_selection": False,
    "permanent_merge_performed": False,
}
metadata_path = output_directory / "checkpoint_sweep_metadata.json"
metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print("__KREA2_RESULT__" + json.dumps(metadata))
unload_runtime(model, transformer, controller, records)
