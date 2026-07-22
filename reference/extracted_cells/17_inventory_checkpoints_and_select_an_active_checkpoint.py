import json
import os
import re
import subprocess
from pathlib import Path

production_directory = PATHS["checkpoints"] / USER_CONFIG["run_name"]
smoke_directory = PATHS["smoke_checkpoints"] / f"{USER_CONFIG['run_name']}_smoke"
production_status_path = PATHS["config"] / "production_training_status.json"
smoke_status_path = PATHS["config"] / "training_smoke_test_result.json"

if production_directory.is_dir() and list(production_directory.glob("*.safetensors")):
    selected_run_directory = production_directory
    selected_run_name = USER_CONFIG["run_name"]
    source_kind = "production"
    process_status = (
        json.loads(production_status_path.read_text(encoding="utf-8"))
        if production_status_path.is_file()
        else {}
    )
elif smoke_directory.is_dir() and list(smoke_directory.glob("*.safetensors")):
    selected_run_directory = smoke_directory
    selected_run_name = f"{USER_CONFIG['run_name']}_smoke"
    source_kind = "smoke"
    process_status = (
        json.loads(smoke_status_path.read_text(encoding="utf-8"))
        if smoke_status_path.is_file()
        else {}
    )
else:
    raise RuntimeError("No production or smoke-test LoRA checkpoints were found.")

inventory_script = r"""
import hashlib
import json
import os
import re
from pathlib import Path
import torch
from safetensors import safe_open

run_directory = Path(os.environ["KREA2_RUN_DIRECTORY"])
run_name = os.environ["KREA2_RUN_NAME"]
configured_steps = int(os.environ["KREA2_CONFIGURED_STEPS"])
source_kind = os.environ["KREA2_SOURCE_KIND"]
process_status = json.loads(os.environ["KREA2_PROCESS_STATUS_JSON"])


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


records = []
reference_keys = None
reference_shapes = None
for path in sorted(run_directory.glob("*.safetensors")):
    final_name = path.stem == run_name
    match = re.fullmatch(re.escape(run_name) + r"_(\d+)", path.stem)
    if not final_name and match is None:
        continue
    step = configured_steps if final_name else int(match.group(1))
    tensor_count = 0
    parameter_count = 0
    nonfinite_count = 0
    ranks = set()
    keys = []
    shapes = {}
    metadata = {}
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        keys = list(checkpoint.keys())
        for key in keys:
            tensor = checkpoint.get_tensor(key)
            shapes[key] = list(tensor.shape)
            tensor_count += 1
            parameter_count += int(tensor.numel())
            nonfinite_count += int((~torch.isfinite(tensor.float())).sum().item())
            if key.endswith(".lora_A.weight") and tensor.ndim == 2:
                ranks.add(int(tensor.shape[0]))
    if nonfinite_count != 0:
        raise RuntimeError(f"Checkpoint contains non-finite parameters: {path}")
    if len(ranks) != 1:
        raise RuntimeError(f"Unable to infer one LoRA rank from {path}: {sorted(ranks)}")
    if reference_keys is None:
        reference_keys = keys
        reference_shapes = shapes
    else:
        if keys != reference_keys or shapes != reference_shapes:
            raise RuntimeError(f"Checkpoint schema differs from the first checkpoint: {path}")
    records.append({
        "step": step,
        "path": str(path),
        "filename": path.name,
        "is_final_name": final_name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "nonfinite_parameter_count": nonfinite_count,
        "rank": next(iter(ranks)),
        "metadata": metadata,
    })

if not records:
    raise RuntimeError("No valid LoRA checkpoint files were discovered.")

records.sort(key=lambda item: (item["step"], item["is_final_name"]))
final_records = [record for record in records if record["is_final_name"]]
numbered_records = [record for record in records if not record["is_final_name"]]
latest_numbered_step = max((record["step"] for record in numbered_records), default=None)

production_process_completed = (
    source_kind == "production"
    and process_status.get("status") == "completed_process"
    and process_status.get("process_return_code") == 0
)
smoke_process_completed = (
    source_kind == "smoke"
    and process_status.get("status") == "passed"
)
configured_step_reached = latest_numbered_step is not None and latest_numbered_step >= configured_steps
training_complete = bool(
    configured_step_reached
    or (production_process_completed and final_records)
    or smoke_process_completed
)

completion_evidence = []
if configured_step_reached:
    completion_evidence.append("numbered_checkpoint_reached_configured_step")
if production_process_completed and final_records:
    completion_evidence.append("production_process_returned_zero_with_final_checkpoint")
if smoke_process_completed:
    completion_evidence.append("smoke_process_passed")
if not completion_evidence:
    completion_evidence.append("no_independent_completion_evidence")

excluded_untrusted_final_files = []
trusted_records = records
if final_records and not training_complete and numbered_records:
    trusted_records = numbered_records
    excluded_untrusted_final_files = [record["path"] for record in final_records]

step_map = {}
for record in trusted_records:
    step_map[record["step"]] = record
unique_records = [step_map[step] for step in sorted(step_map)]

if not unique_records:
    raise RuntimeError("No trusted checkpoint records remain after completion-state validation.")

optimizer_files = [path for path in run_directory.rglob("optimizer.pt") if path.is_file()]
result = {
    "source_kind": source_kind,
    "run_name": run_name,
    "run_directory": str(run_directory),
    "configured_steps": configured_steps,
    "training_complete": training_complete,
    "completion_evidence": completion_evidence,
    "process_status": process_status,
    "checkpoint_count": len(unique_records),
    "checkpoint_steps": [record["step"] for record in unique_records],
    "checkpoints": unique_records,
    "excluded_untrusted_final_files": excluded_untrusted_final_files,
    "optimizer_state_present": bool(optimizer_files),
    "optimizer_files": [str(path) for path in optimizer_files],
    "final_quality_claim_allowed": source_kind == "production" and training_complete,
}
print(json.dumps(result))
"""

environment = os.environ.copy()
environment["KREA2_RUN_DIRECTORY"] = str(selected_run_directory)
environment["KREA2_RUN_NAME"] = selected_run_name
environment["KREA2_CONFIGURED_STEPS"] = str(
    USER_CONFIG["training_steps"] if source_kind == "production" else USER_CONFIG["smoke_test_steps"]
)
environment["KREA2_SOURCE_KIND"] = source_kind
environment["KREA2_PROCESS_STATUS_JSON"] = json.dumps(process_status)
result = subprocess.run(
    [str(PATHS["venv_python"]), "-c", inventory_script],
    cwd=str(PATHS["ai_toolkit"]),
    env=environment,
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    raise RuntimeError(f"Checkpoint inventory failed.\n{result.stdout}\n{result.stderr}")
inventory = json.loads(result.stdout.strip().splitlines()[-1])

mode = USER_CONFIG["checkpoint_selection_mode"]
if mode == "manual":
    step = USER_CONFIG["manual_checkpoint_step"]
    if step is None:
        raise RuntimeError("manual_checkpoint_step must be set when checkpoint_selection_mode is manual.")
    candidates = [record for record in inventory["checkpoints"] if record["step"] == step]
    if len(candidates) != 1:
        raise RuntimeError(f"Manual checkpoint step {step} was not found exactly once.")
    selected = candidates[0]
elif mode == "latest":
    selected = inventory["checkpoints"][-1]
else:
    final_candidates = [
        record
        for record in inventory["checkpoints"]
        if record["is_final_name"] and inventory["training_complete"]
    ]
    selected = final_candidates[-1] if final_candidates else inventory["checkpoints"][-1]

inventory_path = PATHS["config"] / "checkpoint_inventory.json"
selection_path = PATHS["config"] / "active_checkpoint_selection.json"
inventory_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
selection = {
    "selection_mode": mode,
    "source_kind": inventory["source_kind"],
    "training_complete": inventory["training_complete"],
    "completion_evidence": inventory["completion_evidence"],
    "final_quality_claim_allowed": inventory["final_quality_claim_allowed"],
    "checkpoint_step": selected["step"],
    "checkpoint_path": selected["path"],
    "checkpoint_sha256": selected["sha256"],
    "rank": selected["rank"],
    "adapter_name": USER_CONFIG["primary_adapter_name"],
    "adapter_scale": USER_CONFIG["primary_adapter_scale"],
    "pipeline_smoke_test_only": not inventory["final_quality_claim_allowed"],
    "permanent_merge_allowed": False,
}
selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in inventory.items() if key != "checkpoints"}, indent=2))
print(json.dumps(selection, indent=2))
print(f"Checkpoint inventory: {inventory_path}")
print(f"Active checkpoint selection: {selection_path}")
