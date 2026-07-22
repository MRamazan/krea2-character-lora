import json
from pathlib import Path
from google.colab import files

export_directory = PATHS["exports"] / USER_CONFIG["run_name"]
manifest_path = export_directory / "resume_manifest.json"
if not manifest_path.is_file():
    raise RuntimeError("The export manifest is missing. Run the packaging cell first.")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for package in manifest["packages"]:
    print(f"{package['filename']} | {package['size_bytes'] / (1024 ** 3):.3f} GiB | {package['sha256']}")
    if USER_CONFIG["auto_download_exports"]:
        files.download(package["path"])
if not USER_CONFIG["auto_download_exports"]:
    print("Automatic browser downloads are disabled. Set auto_download_exports to true or download the printed files manually.")