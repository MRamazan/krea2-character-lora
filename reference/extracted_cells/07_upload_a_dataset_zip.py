import json
import shutil
from pathlib import Path
from google.colab import files

upload_directory = PATHS["dataset"] / "uploads"
if upload_directory.exists():
    shutil.rmtree(upload_directory)
upload_directory.mkdir(parents=True, exist_ok=False)

uploaded = files.upload()
zip_names = [name for name in uploaded if name.lower().endswith(".zip")]
if len(uploaded) != 1 or len(zip_names) != 1:
    raise RuntimeError("Upload exactly one ZIP archive and no additional files.")

archive_name = zip_names[0]
archive_path = upload_directory / archive_name
archive_path.write_bytes(uploaded[archive_name])
record = {
    "archive_name": archive_name,
    "archive_path": str(archive_path),
    "size_bytes": archive_path.stat().st_size,
}
record_path = PATHS["dataset_audit"] / "dataset_upload_manifest.json"
record_path.parent.mkdir(parents=True, exist_ok=True)
record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))