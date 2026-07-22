import json
from pathlib import Path

records = []
names = {USER_CONFIG["primary_adapter_name"]}
for item in USER_CONFIG["additional_loras"]:
    required = {"name", "path", "scale"}
    missing = required - set(item)
    if missing:
        raise RuntimeError(f"Additional LoRA declaration is missing fields: {sorted(missing)}")
    if item["name"] in names:
        raise RuntimeError(f"Duplicate adapter name: {item['name']}")
    names.add(item["name"])
    path = Path(item["path"])
    if not path.is_file():
        raise RuntimeError(f"Additional LoRA file is missing: {path}")
    records.append({
        "name": item["name"],
        "path": str(path),
        "scale": float(item["scale"]),
        "alpha": item.get("alpha"),
        "exists": True,
    })
result = {"additional_lora_count": len(records), "records": records, "permanent_merge_allowed": False}
result_path = PATHS["config"] / "additional_lora_validation.json"
result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))