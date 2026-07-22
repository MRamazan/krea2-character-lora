import json
import os
import shutil
import zipfile
from pathlib import Path
from PIL import Image

upload_manifest = json.loads((PATHS["dataset_audit"] / "dataset_upload_manifest.json").read_text(encoding="utf-8"))
archive_path = Path(upload_manifest["archive_path"])
raw_directory = PATHS["dataset_raw"]
training_directory = PATHS["dataset_training"]

for directory in [raw_directory, training_directory]:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=False)

raw_root = raw_directory.resolve()
with zipfile.ZipFile(archive_path) as archive:
    for member in archive.infolist():
        destination = (raw_directory / member.filename).resolve()
        try:
            destination.relative_to(raw_root)
        except ValueError as error:
            raise RuntimeError(f"Unsafe ZIP path detected: {member.filename}") from error
    archive.extractall(raw_directory)

allowed_extensions = {".png", ".jpg", ".jpeg"}
unsupported_extensions = {".webp", ".bmp", ".tif", ".tiff", ".gif"}
all_files = [path for path in raw_directory.rglob("*") if path.is_file()]
unsupported_images = [path for path in all_files if path.suffix.lower() in unsupported_extensions]
if unsupported_images:
    raise RuntimeError("Unsupported training image formats were found: " + ", ".join(str(path) for path in unsupported_images))

images = [path for path in all_files if path.suffix.lower() in allowed_extensions]
captions = [path for path in all_files if path.suffix.lower() == ".txt"]

image_map = {}
caption_map = {}
for path in images:
    relative_key = path.relative_to(raw_directory).with_suffix("").as_posix()
    if relative_key in image_map:
        raise RuntimeError(f"Duplicate image key detected: {relative_key}")
    image_map[relative_key] = path
for path in captions:
    relative_key = path.relative_to(raw_directory).with_suffix("").as_posix()
    if relative_key in caption_map:
        raise RuntimeError(f"Duplicate caption key detected: {relative_key}")
    caption_map[relative_key] = path

missing_captions = sorted(set(image_map) - set(caption_map))
missing_images = sorted(set(caption_map) - set(image_map))
if missing_captions or missing_images:
    raise RuntimeError(f"Unmatched files detected. Missing captions: {missing_captions}. Missing images: {missing_images}.")

pair_count = len(image_map)
if pair_count < USER_CONFIG["minimum_pair_count"]:
    raise RuntimeError(f"Dataset contains {pair_count} pairs, below the configured minimum of {USER_CONFIG['minimum_pair_count']}.")
if USER_CONFIG["expected_pair_count"] is not None and pair_count != USER_CONFIG["expected_pair_count"]:
    raise RuntimeError(f"Dataset contains {pair_count} pairs, expected {USER_CONFIG['expected_pair_count']}.")

records = []
for index, source_key in enumerate(sorted(image_map), start=1):
    source_image = image_map[source_key]
    source_caption = caption_map[source_key]
    with Image.open(source_image) as image:
        image.verify()
    with Image.open(source_image) as image:
        width, height = image.size
        mode = image.mode
    caption = source_caption.read_text(encoding="utf-8").strip()
    if not caption:
        raise RuntimeError(f"Caption is empty: {source_caption}")
    canonical_id = f"{index:06d}"
    normalized_extension = source_image.suffix.lower()
    target_image = training_directory / f"{canonical_id}{normalized_extension}"
    target_caption = training_directory / f"{canonical_id}.txt"
    shutil.copy2(source_image, target_image)
    target_caption.write_text(caption, encoding="utf-8")
    records.append({
        "index": index,
        "canonical_id": canonical_id,
        "source_key": source_key,
        "source_image": str(source_image),
        "source_caption": str(source_caption),
        "image": str(target_image),
        "caption": str(target_caption),
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "mode": mode,
        "text": caption,
    })

manifest = {
    "pair_count": pair_count,
    "expected_pair_count": USER_CONFIG["expected_pair_count"],
    "minimum_pair_count": USER_CONFIG["minimum_pair_count"],
    "raw_directory": str(raw_directory),
    "training_directory": str(training_directory),
    "canonicalization": "sorted relative source key to six-digit flat identifier",
    "records": records,
}
manifest_path = PATHS["dataset_audit"] / "dataset_validation_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Validated image-caption pairs: {pair_count}")
print(f"Canonical training directory: {training_directory}")
print(f"Validation manifest: {manifest_path}")