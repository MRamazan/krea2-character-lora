from __future__ import annotations

import math
import shutil
import statistics
import textwrap
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .configuration import DatasetConfig
from .errors import DatasetError
from .hashing import sha256_bytes, sha256_file, sha256_text
from .manifests import read_json, write_json_atomic
from .paths import ProjectPaths
from .types import DatasetResult


@dataclass(slots=True)
class DatasetIssue:
    kind: str
    severity: str
    message: str
    keys: list[str]


def safe_extract(zip_path: str | Path, destination: Path) -> None:
    archive_path = Path(zip_path)
    if not archive_path.is_file():
        raise DatasetError(f"The dataset archive does not exist: {archive_path}")
    if archive_path.suffix.lower() != ".zip":
        raise DatasetError("The dataset archive must be a single ZIP file.")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            resolved = (destination / member.filename).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise DatasetError(f"Unsafe archive path detected: {member.filename}") from error
        archive.extractall(destination)


def discover_pairs(
    raw_directory: Path, accepted_extensions: tuple[str, ...]
) -> tuple[dict[str, Path], dict[str, Path], list[DatasetIssue]]:
    accepted = {value.lower() for value in accepted_extensions}
    unsupported_known = {".webp", ".bmp", ".tif", ".tiff", ".gif"} - accepted
    issues: list[DatasetIssue] = []
    all_files = [path for path in raw_directory.rglob("*") if path.is_file()]
    unsupported = sorted(
        path.relative_to(raw_directory).as_posix()
        for path in all_files
        if path.suffix.lower() in unsupported_known
    )
    if unsupported:
        issues.append(
            DatasetIssue(
                kind="unsupported_image_format",
                severity="error",
                message="Unsupported image formats were found. Convert them to PNG or JPEG.",
                keys=unsupported,
            )
        )
    image_map: dict[str, Path] = {}
    caption_map: dict[str, Path] = {}
    duplicate_image_keys: list[str] = []
    duplicate_caption_keys: list[str] = []
    for path in sorted(all_files, key=lambda item: item.as_posix()):
        suffix = path.suffix.lower()
        key = path.relative_to(raw_directory).with_suffix("").as_posix()
        if suffix in accepted:
            if key in image_map:
                duplicate_image_keys.append(key)
            else:
                image_map[key] = path
        elif suffix == ".txt":
            if key in caption_map:
                duplicate_caption_keys.append(key)
            else:
                caption_map[key] = path
    if duplicate_image_keys:
        issues.append(
            DatasetIssue(
                kind="duplicate_image_key",
                severity="error",
                message="Multiple images share the same relative name without extension.",
                keys=sorted(set(duplicate_image_keys)),
            )
        )
    if duplicate_caption_keys:
        issues.append(
            DatasetIssue(
                kind="duplicate_caption_key",
                severity="error",
                message="Multiple captions share the same relative name without extension.",
                keys=sorted(set(duplicate_caption_keys)),
            )
        )
    missing_captions = sorted(set(image_map) - set(caption_map))
    orphan_captions = sorted(set(caption_map) - set(image_map))
    if missing_captions:
        issues.append(
            DatasetIssue(
                kind="missing_caption",
                severity="error",
                message="Images without a matching caption file were found.",
                keys=missing_captions,
            )
        )
    if orphan_captions:
        issues.append(
            DatasetIssue(
                kind="orphan_caption",
                severity="error",
                message="Caption files without a matching image were found.",
                keys=orphan_captions,
            )
        )
    return image_map, caption_map, issues


def _difference_hash(path: Path, width: int = 16, height: int = 16) -> int:
    from PIL import Image

    with Image.open(path) as opened:
        resized = opened.convert("L").resize((width + 1, height), Image.Resampling.LANCZOS)
        pixels = list(resized.tobytes())
    value = 0
    for row in range(height):
        offset = row * (width + 1)
        for column in range(width):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _read_image_properties(path: Path) -> tuple[int, int, str]:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as opened:
            opened.verify()
        with Image.open(path) as opened:
            width, height = opened.size
            mode = opened.mode
    except (UnidentifiedImageError, OSError) as error:
        raise DatasetError(f"The image could not be decoded: {path.name}") from error
    return width, height, mode


def build_records(
    image_map: dict[str, Path],
    caption_map: dict[str, Path],
    training_directory: Path,
    accepted_extensions: tuple[str, ...],
    trigger_word: str,
) -> tuple[list[dict], list[DatasetIssue]]:
    issues: list[DatasetIssue] = []
    if training_directory.exists():
        shutil.rmtree(training_directory)
    training_directory.mkdir(parents=True, exist_ok=False)
    records: list[dict] = []
    corrupt_keys: list[str] = []
    empty_keys: list[str] = []
    matched_keys = sorted(set(image_map) & set(caption_map))
    for index, source_key in enumerate(matched_keys, start=1):
        source_image = image_map[source_key]
        source_caption = caption_map[source_key]
        try:
            width, height, mode = _read_image_properties(source_image)
        except DatasetError:
            corrupt_keys.append(source_key)
            continue
        caption_text = source_caption.read_text(encoding="utf-8").strip()
        if not caption_text:
            empty_keys.append(source_key)
            continue
        canonical_id = f"{index:06d}"
        extension = source_image.suffix.lower()
        target_image = training_directory / f"{canonical_id}{extension}"
        target_caption = training_directory / f"{canonical_id}.txt"
        shutil.copy2(source_image, target_image)
        target_caption.write_text(caption_text, encoding="utf-8")
        records.append(
            {
                "index": index,
                "canonical_id": canonical_id,
                "source_key": source_key,
                "source_image": str(source_image),
                "image": str(target_image),
                "caption": str(target_caption),
                "width": width,
                "height": height,
                "aspect_ratio": width / height,
                "mode": mode,
                "text": caption_text,
                "trigger_count": caption_text.count(trigger_word) if trigger_word else 0,
                "image_sha256": sha256_file(target_image),
                "caption_sha256": sha256_text(caption_text),
                "dhash": _difference_hash(target_image),
            }
        )
    if corrupt_keys:
        issues.append(
            DatasetIssue(
                kind="corrupt_image",
                severity="error",
                message="One or more images could not be decoded.",
                keys=sorted(corrupt_keys),
            )
        )
    if empty_keys:
        issues.append(
            DatasetIssue(
                kind="empty_caption",
                severity="error",
                message="One or more caption files are empty.",
                keys=sorted(empty_keys),
            )
        )
    return records, issues


def audit_trigger(
    records: list[dict], config: DatasetConfig, backup_directory: Path
) -> tuple[list[dict], list[DatasetIssue]]:
    issues: list[DatasetIssue] = []
    trigger_word = config.trigger_word
    missing = [record for record in records if record["trigger_count"] == 0]
    modified: list[str] = []
    if missing and config.auto_prefix_missing_trigger:
        if backup_directory.exists():
            shutil.rmtree(backup_directory)
        backup_directory.mkdir(parents=True, exist_ok=False)
        for record in missing:
            caption_path = Path(record["caption"])
            backup_path = backup_directory / f"{record['canonical_id']}.txt"
            shutil.copy2(caption_path, backup_path)
            original = caption_path.read_text(encoding="utf-8").strip()
            updated = f"{trigger_word}, {original}"
            caption_path.write_text(updated, encoding="utf-8")
            record["text"] = updated
            record["trigger_count"] = updated.count(trigger_word)
            record["caption_sha256"] = sha256_text(updated)
            modified.append(record["canonical_id"])
        missing = [record for record in records if record["trigger_count"] == 0]
    status = "passed"
    if missing:
        missing_keys = [record["canonical_id"] for record in missing]
        if config.caption_trigger_policy == "require":
            status = "failed"
            issues.append(
                DatasetIssue(
                    kind="missing_trigger",
                    severity="error",
                    message=f"The trigger word '{trigger_word}' is missing from captions.",
                    keys=missing_keys,
                )
            )
        elif config.caption_trigger_policy == "warn":
            status = "warning"
            issues.append(
                DatasetIssue(
                    kind="missing_trigger",
                    severity="warning",
                    message=f"The trigger word '{trigger_word}' is missing from captions.",
                    keys=missing_keys,
                )
            )
    audit = {
        "trigger_word": trigger_word,
        "policy": config.caption_trigger_policy,
        "auto_prefix_missing_trigger": config.auto_prefix_missing_trigger,
        "status": status,
        "missing_caption_count": len(missing),
        "modified_captions": modified,
        "backup_directory": str(backup_directory) if modified else None,
    }
    for record in records:
        record["trigger_audit"] = record["trigger_count"] > 0
    return [audit], issues


def audit_duplicates(records: list[dict], config: DatasetConfig) -> tuple[dict, list[DatasetIssue]]:
    issues: list[DatasetIssue] = []
    exact_groups: dict[str, list[str]] = {}
    for record in records:
        exact_groups.setdefault(record["image_sha256"], []).append(record["source_key"])
    exact_duplicate_groups = [group for group in exact_groups.values() if len(group) > 1]
    threshold = config.near_duplicate_hamming_threshold
    near_duplicates: list[dict] = []
    for left_index in range(len(records)):
        for right_index in range(left_index + 1, len(records)):
            distance = _hamming_distance(
                records[left_index]["dhash"], records[right_index]["dhash"]
            )
            if distance <= threshold:
                near_duplicates.append(
                    {
                        "left": records[left_index]["source_key"],
                        "right": records[right_index]["source_key"],
                        "distance": distance,
                    }
                )
    near_duplicates.sort(key=lambda item: (item["distance"], item["left"], item["right"]))
    if exact_duplicate_groups:
        severity = "error" if config.fail_on_exact_duplicates else "warning"
        issues.append(
            DatasetIssue(
                kind="exact_duplicate",
                severity=severity,
                message="Exact duplicate images were found.",
                keys=[",".join(group) for group in exact_duplicate_groups],
            )
        )
    if near_duplicates:
        issues.append(
            DatasetIssue(
                kind="near_duplicate",
                severity="warning",
                message="Near-duplicate image candidates were found.",
                keys=[f"{item['left']}~{item['right']}" for item in near_duplicates],
            )
        )
    audit = {
        "near_duplicate_hamming_threshold": threshold,
        "exact_duplicate_groups": exact_duplicate_groups,
        "exact_duplicate_count": len(exact_duplicate_groups),
        "near_duplicate_candidates": near_duplicates,
        "near_duplicate_candidate_count": len(near_duplicates),
    }
    return audit, issues


def compute_fingerprint(records: list[dict]) -> str:
    payload = "".join(
        f"{record['source_key']}:{record['image_sha256']}:{record['caption_sha256']}"
        for record in sorted(records, key=lambda item: item["source_key"])
    )
    return sha256_bytes(payload.encode("utf-8"))


def _dimension_summary(values: list[float]) -> dict:
    return {
        "minimum": min(values),
        "maximum": max(values),
        "median": statistics.median(values),
    }


def _fatal(issues: list[DatasetIssue]) -> list[DatasetIssue]:
    return [issue for issue in issues if issue.severity == "error"]


def prepare_dataset(
    zip_path: str | Path, config: DatasetConfig, paths: ProjectPaths
) -> DatasetResult:
    config.validate()
    raw_directory = paths.dataset_raw
    training_directory = paths.dataset_training
    audit_directory = paths.dataset_audit
    audit_directory.mkdir(parents=True, exist_ok=True)
    safe_extract(zip_path, raw_directory)
    accepted = config.normalized_extensions()
    image_map, caption_map, discovery_issues = discover_pairs(raw_directory, accepted)
    records, record_issues = build_records(
        image_map, caption_map, training_directory, accepted, config.trigger_word
    )
    trigger_records, trigger_issues = audit_trigger(
        records, config, paths.dataset_active / "caption_backups"
    )
    duplicate_audit, duplicate_issues = audit_duplicates(records, config)
    issues = discovery_issues + record_issues + trigger_issues + duplicate_issues
    pair_count = len(records)
    if pair_count < config.minimum_pair_count:
        issues.append(
            DatasetIssue(
                kind="insufficient_pairs",
                severity="error",
                message=(
                    f"The dataset contains {pair_count} pairs, "
                    f"below the minimum of {config.minimum_pair_count}."
                ),
                keys=[],
            )
        )
    if config.expected_pair_count is not None and pair_count != config.expected_pair_count:
        issues.append(
            DatasetIssue(
                kind="unexpected_pair_count",
                severity="error",
                message=(
                    f"The dataset contains {pair_count} pairs, "
                    f"but {config.expected_pair_count} were expected."
                ),
                keys=[],
            )
        )
    fingerprint = compute_fingerprint(records)
    widths = [record["width"] for record in records] or [0]
    heights = [record["height"] for record in records] or [0]
    ratios = [record["aspect_ratio"] for record in records] or [0.0]
    manifest = {
        "schema_version": 1,
        "trigger_word": config.trigger_word,
        "caption_trigger_policy": config.caption_trigger_policy,
        "auto_prefix_missing_trigger": config.auto_prefix_missing_trigger,
        "accepted_image_extensions": list(accepted),
        "minimum_pair_count": config.minimum_pair_count,
        "expected_pair_count": config.expected_pair_count,
        "pair_count": pair_count,
        "raw_directory": str(raw_directory),
        "training_directory": str(training_directory),
        "dataset_fingerprint_sha256": fingerprint,
        "dimensions": {
            "width": _dimension_summary(widths),
            "height": _dimension_summary(heights),
            "aspect_ratio": _dimension_summary(ratios),
        },
        "trigger_audit": trigger_records[0],
        "duplicate_audit": duplicate_audit,
        "issues": [asdict(issue) for issue in issues],
        "gallery": {
            "columns": config.gallery_columns,
            "thumbnail_size": config.gallery_thumbnail_size,
            "page_size": config.gallery_page_size,
        },
        "records": records,
    }
    write_json_atomic(paths.dataset_manifest, manifest)
    write_json_atomic(
        paths.dataset_fingerprint,
        {
            "dataset_fingerprint_sha256": fingerprint,
            "pair_count": pair_count,
            "records": [
                {
                    "canonical_id": record["canonical_id"],
                    "source_key": record["source_key"],
                    "image_sha256": record["image_sha256"],
                    "caption_sha256": record["caption_sha256"],
                }
                for record in records
            ],
        },
    )
    fatal = _fatal(issues)
    if fatal:
        summary = "; ".join(f"{issue.kind} ({len(issue.keys)})" for issue in fatal)
        raise DatasetError(
            "The dataset failed validation. Review the manifest and resolve these issues: "
            f"{summary}. Manifest: {paths.dataset_manifest}"
        )
    return DatasetResult(
        workspace=paths.root,
        manifest_path=paths.dataset_manifest,
        trigger_word=config.trigger_word,
        details=manifest,
    )


def load_dataset(paths: ProjectPaths) -> DatasetResult:
    if not paths.dataset_manifest.is_file():
        raise DatasetError(
            f"No dataset manifest was found at {paths.dataset_manifest}. Run prepare_dataset first."
        )
    manifest = read_json(paths.dataset_manifest)
    return DatasetResult(
        workspace=paths.root,
        manifest_path=paths.dataset_manifest,
        trigger_word=manifest["trigger_word"],
        details=manifest,
    )


def highlight_trigger(text: str, trigger_word: str) -> str:
    if not trigger_word or trigger_word not in text:
        return text
    return text.replace(trigger_word, f"[{trigger_word}]")


def render_summary(details: dict) -> str:
    dimensions = details["dimensions"]
    duplicates = details["duplicate_audit"]
    lines = [
        f"Trigger word: {details['trigger_word']}",
        f"Pairs: {details['pair_count']}",
        f"Fingerprint: {details['dataset_fingerprint_sha256'][:16]}",
        (
            "Width: "
            f"{dimensions['width']['minimum']}-{dimensions['width']['maximum']} "
            f"(median {dimensions['width']['median']})"
        ),
        (
            "Height: "
            f"{dimensions['height']['minimum']}-{dimensions['height']['maximum']} "
            f"(median {dimensions['height']['median']})"
        ),
        f"Trigger audit: {details['trigger_audit']['status']}",
        f"Exact duplicate groups: {duplicates['exact_duplicate_count']}",
        f"Near-duplicate candidates: {duplicates['near_duplicate_candidate_count']}",
        f"Issues: {len(details['issues'])}",
    ]
    return "\n".join(lines)


def render_caption_audit(details: dict) -> str:
    header = f"{'id':>7}  {'triggers':>8}  caption"
    rows = [header, "-" * len(header)]
    for record in details["records"]:
        caption = record["text"]
        if len(caption) > 80:
            caption = caption[:77] + "..."
        rows.append(
            f"{record['canonical_id']:>7}  {record['trigger_count']:>8}  "
            f"{highlight_trigger(caption, details['trigger_word'])}"
        )
    return "\n".join(rows)


def render_issues(details: dict) -> str:
    issues = details["issues"]
    if not issues:
        return "No dataset issues were detected."
    rows = []
    for issue in issues:
        keys = ", ".join(issue["keys"][:8])
        suffix = "" if len(issue["keys"]) <= 8 else f" (+{len(issue['keys']) - 8} more)"
        rows.append(f"[{issue['severity'].upper()}] {issue['kind']}: {issue['message']}")
        if keys:
            rows.append(f"    {keys}{suffix}")
    return "\n".join(rows)


def gallery_pages(details: dict, page_size: int) -> list[list[dict]]:
    records = details["records"]
    total = math.ceil(len(records) / page_size) if records else 0
    return [records[index * page_size : (index + 1) * page_size] for index in range(total)]


def render_gallery(
    details: dict,
    columns: int,
    thumbnail_size: int,
    show_filename: bool,
    show_dimensions: bool,
    show_caption: bool,
    highlight: bool,
    page_size: int,
    page: int | None,
) -> None:
    try:
        from IPython.display import display
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print(render_caption_audit(details))
        return
    pages = gallery_pages(details, page_size)
    if not pages:
        print("The dataset gallery is empty.")
        return
    selected = pages if page is None else [pages[page]]
    font = ImageFont.load_default()
    cell_width = thumbnail_size + 24
    caption_height = 120 if show_caption else 40
    cell_height = thumbnail_size + caption_height
    for records in selected:
        rows = math.ceil(len(records) / columns)
        canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
        draw = ImageDraw.Draw(canvas)
        for local_index, record in enumerate(records):
            column = local_index % columns
            row = local_index // columns
            x = column * cell_width
            y = row * cell_height
            with Image.open(record["image"]) as opened:
                thumbnail = opened.convert("RGB")
                thumbnail.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
                canvas.paste(thumbnail, (x + 12, y + 12))
            text_y = y + thumbnail_size + 16
            if show_filename:
                draw.text(
                    (x + 12, text_y),
                    f"{Path(record['image']).name} | {record['source_key']}",
                    fill="black",
                    font=font,
                )
                text_y += 14
            if show_dimensions:
                draw.text(
                    (x + 12, text_y),
                    f"{record['width']}x{record['height']} r{record['aspect_ratio']:.3f}",
                    fill="black",
                    font=font,
                )
                text_y += 14
            if show_caption:
                caption = record["text"]
                if highlight:
                    caption = highlight_trigger(caption, details["trigger_word"])
                for line in textwrap.wrap(caption, width=42)[:5]:
                    draw.text((x + 12, text_y), line, fill="black", font=font)
                    text_y += 14
        display(canvas)
