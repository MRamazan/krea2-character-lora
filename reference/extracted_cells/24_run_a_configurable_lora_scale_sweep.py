import json
import os
import subprocess
from pathlib import Path
from IPython.display import display
from PIL import Image

if not USER_CONFIG["run_inference"] or not USER_CONFIG["run_scale_sweep"]:
    print("LoRA scale sweep skipped by configuration.")
else:
    output_directory = PATHS["inference"] / "scale_sweep"
    output_directory.mkdir(parents=True, exist_ok=True)
    script = r"""
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw
from toolkit.config_modules import GenerateImageConfig
from runtime_helpers.krea2_runtime import load_runtime, sha256_file, unload_runtime

root = Path(os.environ["KREA2_PROJECT_ROOT"])
config = json.loads((root / "config" / "user_configuration.json").read_text(encoding="utf-8"))
selection = json.loads((root / "config" / "active_checkpoint_selection.json").read_text(encoding="utf-8"))
assets = json.loads((root / "config" / "inference_asset_manifest.json").read_text(encoding="utf-8"))
output_directory = Path(os.environ["KREA2_OUTPUT_DIRECTORY"])
adapter_specs = [{
    "name": config["primary_adapter_name"],
    "path": selection["checkpoint_path"],
    "scale": config["primary_adapter_scale"],
    "alpha": config["lora_alpha"],
}] + config["additional_loras"]
model, transformer, controller, records = load_runtime(assets, adapter_specs, dtype=config["training_dtype"], max_text_length=config["max_text_length"])
image_records = []
paths_by_case = {index: [] for index in range(len(config["evaluation_prompts"]))}
for scale in config["scale_sweep"]:
    scales = {config["primary_adapter_name"]: float(scale)}
    for item in config["additional_loras"]:
        scales[item["name"]] = float(item.get("scale", 1.0))
    controller.set_scales(scales)
    configurations = []
    expected = []
    for case_index, prompt in enumerate(config["evaluation_prompts"]):
        seed = int(config["evaluation_seeds"][case_index % len(config["evaluation_seeds"])])
        scale_label = str(scale).replace(".", "p")
        output_path = output_directory / f"case_{case_index + 1:02d}_scale_{scale_label}.png"
        configurations.append(GenerateImageConfig(
            prompt=prompt,
            width=int(config["inference_width"]),
            height=int(config["inference_height"]),
            num_inference_steps=int(config["inference_steps"]),
            guidance_scale=float(config["inference_guidance"]),
            negative_prompt=config["negative_prompt"],
            seed=seed,
            network_multiplier=1.0,
            output_path=str(output_path),
            output_ext="png",
            add_prompt_file=False,
        ))
        expected.append((case_index, prompt, seed, output_path))
    model.generate_images(configurations)
    for case_index, prompt, seed, output_path in expected:
        if not output_path.is_file():
            raise RuntimeError(f"Expected scale-sweep image is missing: {output_path}")
        image_records.append({
            "case_index": case_index + 1,
            "scale": float(scale),
            "prompt": prompt,
            "seed": seed,
            "path": str(output_path),
            "sha256": sha256_file(output_path),
        })
        paths_by_case[case_index].append({"scale": float(scale), "path": str(output_path)})
thumbnail = 384
label_height = 48
grid_paths = []
master_rows = []
for case_index in range(len(config["evaluation_prompts"])):
    items = paths_by_case[case_index]
    grid = Image.new("RGB", (thumbnail * len(items), thumbnail + label_height), "white")
    draw = ImageDraw.Draw(grid)
    for column, item in enumerate(items):
        image = Image.open(item["path"]).convert("RGB").resize((thumbnail, thumbnail), Image.Resampling.LANCZOS)
        grid.paste(image, (column * thumbnail, label_height))
        draw.text((column * thumbnail + 12, 16), f"Scale {item['scale']}", fill="black")
    grid_path = output_directory / f"case_{case_index + 1:02d}_scale_grid.png"
    grid.save(grid_path)
    grid_paths.append(str(grid_path))
    master_rows.append(grid)
master_width = max(row.width for row in master_rows)
master_height = sum(row.height for row in master_rows)
master = Image.new("RGB", (master_width, master_height), "white")
y = 0
for row in master_rows:
    master.paste(row, (0, y))
    y += row.height
master_path = output_directory / "scale_master_grid.png"
master.save(master_path)
metadata = {
    "checkpoint_step": selection["checkpoint_step"],
    "checkpoint_path": selection["checkpoint_path"],
    "scales": [float(scale) for scale in config["scale_sweep"]],
    "prompt_count": len(config["evaluation_prompts"]),
    "image_count": len(image_records),
    "records": image_records,
    "per_prompt_grids": grid_paths,
    "master_grid": str(master_path),
    "permanent_merge_performed": False,
}
metadata_path = output_directory / "scale_sweep_metadata.json"
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
    log_path = PATHS["logs"] / "scale_sweep.log"
    log_path.write_text(result.stdout + result.stderr + f"\nExit code: {result.returncode}\n", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Scale sweep failed.\n{result.stdout}\n{result.stderr}\nComplete log: {log_path}")
    structured = [line for line in result.stdout.splitlines() if line.startswith("__KREA2_RESULT__")]
    if len(structured) != 1:
        raise RuntimeError("The scale sweep returned an invalid structured result.")
    metadata = json.loads(structured[0].removeprefix("__KREA2_RESULT__"))
    print(json.dumps({key: value for key, value in metadata.items() if key != "records"}, indent=2))
    display(Image.open(metadata["master_grid"]).convert("RGB"))