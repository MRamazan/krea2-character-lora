import hashlib
import itertools
import json
from pathlib import Path
from PIL import Image

validation = json.loads((PATHS["dataset_audit"] / "dataset_validation_manifest.json").read_text(encoding="utf-8"))

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def difference_hash(path, width=16, height=16):
    image = Image.open(path).convert("L").resize((width + 1, height), Image.Resampling.LANCZOS)
    pixels = list(image.getdata())
    bits = []
    for row in range(height):
        offset = row * (width + 1)
        for column in range(width):
            bits.append(pixels[offset + column] > pixels[offset + column + 1])
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value

def hamming_distance(left, right):
    return (left ^ right).bit_count()

records = []
for item in validation["records"]:
    image_path = Path(item["image"])
    records.append({
        "path": str(image_path),
        "filename": image_path.name,
        "source_key": item["source_key"],
        "sha256": sha256_file(image_path),
        "dhash": difference_hash(image_path),
    })

exact_groups = {}
for record in records:
    exact_groups.setdefault(record["sha256"], []).append(record["source_key"])
exact_duplicates = [group for group in exact_groups.values() if len(group) > 1]

threshold = int(USER_CONFIG["near_duplicate_hamming_threshold"])
near_duplicates = []
for left, right in itertools.combinations(records, 2):
    distance = hamming_distance(left["dhash"], right["dhash"])
    if distance <= threshold:
        near_duplicates.append({"left": left["source_key"], "right": right["source_key"], "distance": distance})

result = {
    "near_duplicate_hamming_threshold": threshold,
    "exact_duplicate_groups": exact_duplicates,
    "near_duplicate_candidates": sorted(near_duplicates, key=lambda item: item["distance"]),
    "exact_duplicate_count": len(exact_duplicates),
    "near_duplicate_candidate_count": len(near_duplicates),
}
result_path = PATHS["dataset_audit"] / "duplicate_audit.json"
result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

if exact_duplicates and USER_CONFIG["fail_on_exact_duplicates"]:
    raise RuntimeError(f"Exact duplicate image groups were found: {exact_duplicates}")

print(json.dumps(result, indent=2))
print(f"Duplicate audit: {result_path}")