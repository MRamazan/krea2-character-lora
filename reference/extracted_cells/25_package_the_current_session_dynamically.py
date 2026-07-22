import hashlib
import json
import math
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

export_directory = PATHS["exports"] / USER_CONFIG["run_name"]
if export_directory.exists():
    shutil.rmtree(export_directory)
export_directory.mkdir(parents=True, exist_ok=False)

inventory = json.loads((PATHS["config"] / "checkpoint_inventory.json").read_text(encoding="utf-8"))
selection = json.loads((PATHS["config"] / "active_checkpoint_selection.json").read_text(encoding="utf-8"))

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def write_zip(path, entries):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for source, archive_name in entries:
            archive.write(source, arcname=str(archive_name))
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Archive creation failed: {path}")

excluded_roots = {PATHS["models"].resolve(), PATHS["venv"].resolve(), PATHS["ai_toolkit"].resolve(), PATHS["exports"].resolve(), PATHS["checkpoints"].resolve(), PATHS["smoke_checkpoints"].resolve()}
core_entries = []
for path in PROJECT_ROOT.rglob("*"):
    if not path.is_file() or path.is_symlink():
        continue
    resolved = path.resolve()
    if any(str(resolved).startswith(str(root) + "/") or resolved == root for root in excluded_roots):
        continue
    core_entries.append((path, Path("project") / path.relative_to(PROJECT_ROOT)))

core_archive = export_directory / f"{USER_CONFIG['run_name']}_core.zip"
write_zip(core_archive, core_entries)

checkpoint_archives = []
checkpoint_records = inventory["checkpoints"]
chunk_size = USER_CONFIG["checkpoints_per_archive"]
for group_index in range(math.ceil(len(checkpoint_records) / chunk_size)):
    group = checkpoint_records[group_index * chunk_size:(group_index + 1) * chunk_size]
    archive_path = export_directory / f"{USER_CONFIG['run_name']}_checkpoints_{group_index + 1:02d}.zip"
    write_zip(archive_path, [(Path(record["path"]), Path("checkpoints") / Path(record["path"]).name) for record in group])
    checkpoint_archives.append(archive_path)

run_directory = Path(inventory["run_directory"])
training_state_files = [path for path in run_directory.rglob("*") if path.is_file() and path.suffix.lower() != ".safetensors"]
training_state_archive = None
if training_state_files:
    training_state_archive = export_directory / f"{USER_CONFIG['run_name']}_training_state.zip"
    write_zip(training_state_archive, [(path, Path("training_state") / path.relative_to(run_directory)) for path in training_state_files])

selected_source = Path(selection["checkpoint_path"])
selected_copy = export_directory / f"{USER_CONFIG['run_name']}_selected_step_{int(selection['checkpoint_step']):08d}.safetensors"
shutil.copy2(selected_source, selected_copy)
if sha256_file(selected_source) != sha256_file(selected_copy):
    raise RuntimeError("The selected checkpoint copy does not match its source.")

packages = [core_archive, *checkpoint_archives, selected_copy]
if training_state_archive is not None:
    packages.append(training_state_archive)
package_records = [
    {
        "filename": path.name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    for path in packages
]
manifest = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "project_name": USER_CONFIG["project_name"],
    "run_name": USER_CONFIG["run_name"],
    "trigger_word": USER_CONFIG["trigger_word"],
    "source_kind": inventory["source_kind"],
    "training_complete": inventory["training_complete"],
    "final_quality_claim_allowed": inventory["final_quality_claim_allowed"],
    "checkpoint_steps": inventory["checkpoint_steps"],
    "selected_checkpoint": {
        "step": selection["checkpoint_step"],
        "source_path": selection["checkpoint_path"],
        "export_path": str(selected_copy),
        "sha256": sha256_file(selected_copy),
    },
    "packages": package_records,
    "excluded_reproducible_assets": [str(PATHS["models"]), str(PATHS["venv"]), str(PATHS["ai_toolkit"])],
}
manifest_path = export_directory / "resume_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
metadata_archive = export_directory / f"{USER_CONFIG['run_name']}_resume_metadata.zip"
write_zip(metadata_archive, [(manifest_path, Path("resume_manifest.json"))])
package_records.insert(0, {
    "filename": metadata_archive.name,
    "path": str(metadata_archive),
    "size_bytes": metadata_archive.stat().st_size,
    "sha256": sha256_file(metadata_archive),
})
manifest["packages"] = package_records
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(manifest, indent=2, ensure_ascii=False))
print(f"Export directory: {export_directory}")