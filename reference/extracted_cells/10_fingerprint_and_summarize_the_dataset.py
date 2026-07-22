import hashlib
import json
import statistics
from pathlib import Path

validation = json.loads((PATHS["dataset_audit"] / "dataset_validation_manifest.json").read_text(encoding="utf-8"))

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

fingerprint_digest = hashlib.sha256()
records = []
for item in validation["records"]:
    image_path = Path(item["image"])
    caption_path = Path(item["caption"])
    image_hash = sha256_file(image_path)
    caption_hash = sha256_file(caption_path)
    fingerprint_digest.update(item["source_key"].encode("utf-8"))
    fingerprint_digest.update(image_hash.encode("ascii"))
    fingerprint_digest.update(caption_hash.encode("ascii"))
    records.append({
        "canonical_id": item["canonical_id"],
        "source_key": item["source_key"],
        "image_sha256": image_hash,
        "caption_sha256": caption_hash,
        "width": item["width"],
        "height": item["height"],
        "aspect_ratio": item["aspect_ratio"],
    })

widths = [record["width"] for record in records]
heights = [record["height"] for record in records]
ratios = [record["aspect_ratio"] for record in records]
result = {
    "dataset_fingerprint_sha256": fingerprint_digest.hexdigest(),
    "pair_count": len(records),
    "width": {"minimum": min(widths), "maximum": max(widths), "median": statistics.median(widths)},
    "height": {"minimum": min(heights), "maximum": max(heights), "median": statistics.median(heights)},
    "aspect_ratio": {"minimum": min(ratios), "maximum": max(ratios), "median": statistics.median(ratios)},
    "records": records,
}
result_path = PATHS["dataset_audit"] / "dataset_fingerprint.json"
result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
print(f"Dataset fingerprint: {result_path}")