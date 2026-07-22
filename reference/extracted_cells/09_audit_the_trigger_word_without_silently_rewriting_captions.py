import json
import shutil
from pathlib import Path

manifest_path = PATHS["dataset_audit"] / "dataset_validation_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
trigger_word = USER_CONFIG["trigger_word"]
policy = USER_CONFIG["caption_trigger_policy"]
backup_directory = PATHS["dataset"] / "caption_backups"

records = []
missing = []
for item in manifest["records"]:
    caption_path = Path(item["caption"])
    text = caption_path.read_text(encoding="utf-8").strip()
    count = text.count(trigger_word) if trigger_word else 0
    if trigger_word and count == 0:
        missing.append(caption_path)
    records.append({"caption": str(caption_path), "trigger_count": count, "text": text})

modified = []
if trigger_word and missing and USER_CONFIG["auto_prefix_missing_trigger"]:
    if backup_directory.exists():
        shutil.rmtree(backup_directory)
    backup_directory.mkdir(parents=True, exist_ok=False)
    for caption_path in missing:
        backup_path = backup_directory / caption_path.name
        shutil.copy2(caption_path, backup_path)
        original = caption_path.read_text(encoding="utf-8").strip()
        caption_path.write_text(f"{trigger_word}, {original}", encoding="utf-8")
        modified.append(str(caption_path))
    missing = []
    records = []
    for item in manifest["records"]:
        caption_path = Path(item["caption"])
        text = caption_path.read_text(encoding="utf-8").strip()
        records.append({"caption": str(caption_path), "trigger_count": text.count(trigger_word), "text": text})

if trigger_word and missing and policy == "require":
    raise RuntimeError("The trigger word is missing from captions: " + ", ".join(path.name for path in missing))

status = "passed"
if trigger_word and missing and policy == "warn":
    status = "warning"

result = {
    "trigger_word": trigger_word,
    "policy": policy,
    "auto_prefix_missing_trigger": USER_CONFIG["auto_prefix_missing_trigger"],
    "modified_captions": modified,
    "missing_caption_count": len(missing),
    "status": status,
    "records": records,
}
result_path = PATHS["dataset_audit"] / "caption_trigger_audit.json"
result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2, ensure_ascii=False))
print(f"Caption audit: {result_path}")