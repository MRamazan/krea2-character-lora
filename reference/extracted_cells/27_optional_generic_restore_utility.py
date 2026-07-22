import json
import shutil
import zipfile
from pathlib import Path
from google.colab import files

restore_root = Path("/content/krea2_lora_restore").resolve()
if restore_root.exists():
    shutil.rmtree(restore_root)
restore_root.mkdir(parents=True, exist_ok=False)

uploaded = files.upload()
zip_names = sorted(name for name in uploaded if name.lower().endswith(".zip"))
if not zip_names:
    raise RuntimeError("Upload at least one generic session ZIP archive.")

for name in zip_names:
    archive_path = restore_root / name
    archive_path.write_bytes(uploaded[name])
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (restore_root / "extracted" / member.filename).resolve()
            if not str(destination).startswith(str((restore_root / "extracted").resolve())):
                raise RuntimeError(f"Unsafe ZIP path detected: {member.filename}")
        archive.extractall(restore_root / "extracted")

manifests = list((restore_root / "extracted").rglob("resume_manifest.json"))
result = {
    "restore_root": str(restore_root),
    "uploaded_archives": zip_names,
    "resume_manifests": [str(path) for path in manifests],
    "base_models_restored": False,
    "environment_restored": False,
}
print(json.dumps(result, indent=2))