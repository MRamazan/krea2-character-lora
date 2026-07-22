import json
from pathlib import Path

manual_step = USER_CONFIG["manual_checkpoint_step"]
if manual_step is None:
    print("No manual checkpoint step is configured. The current active selection remains unchanged.")
else:
    inventory = json.loads((PATHS["config"] / "checkpoint_inventory.json").read_text(encoding="utf-8"))
    candidates = [record for record in inventory["checkpoints"] if record["step"] == manual_step]
    if len(candidates) != 1:
        raise RuntimeError(f"Manual checkpoint step {manual_step} was not found exactly once.")
    selected = candidates[0]
    selection = {
        "selection_mode": "manual_after_review",
        "source_kind": inventory["source_kind"],
        "training_complete": inventory["training_complete"],
        "final_quality_claim_allowed": inventory["final_quality_claim_allowed"],
        "checkpoint_step": selected["step"],
        "checkpoint_path": selected["path"],
        "checkpoint_sha256": selected["sha256"],
        "rank": selected["rank"],
        "adapter_name": USER_CONFIG["primary_adapter_name"],
        "adapter_scale": USER_CONFIG["primary_adapter_scale"],
        "permanent_merge_allowed": False,
    }
    selection_path = PATHS["config"] / "active_checkpoint_selection.json"
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2))