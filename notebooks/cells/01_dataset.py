# ============================================================
# Dataset
# ============================================================

from google.colab import files

from krea2_character_lora import DatasetConfig

TRIGGER_WORD = "mycharacter"
CAPTION_TRIGGER_POLICY = "require"
AUTO_PREFIX_MISSING_TRIGGER = False
MINIMUM_PAIR_COUNT = 4
EXPECTED_PAIR_COUNT = None
FAIL_ON_EXACT_DUPLICATES = True
NEAR_DUPLICATE_HAMMING_THRESHOLD = 8
ACCEPTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
GALLERY_COLUMNS = 4
GALLERY_THUMBNAIL_SIZE = 320
GALLERY_PAGE_SIZE = 24

uploaded = files.upload()
zip_files = [name for name in uploaded if name.lower().endswith(".zip")]

if len(zip_files) != 1:
    raise RuntimeError("Upload exactly one dataset ZIP file.")

DATASET_ZIP = f"/content/{zip_files[0]}"

dataset_config = DatasetConfig(
    trigger_word=TRIGGER_WORD,
    caption_trigger_policy=CAPTION_TRIGGER_POLICY,
    auto_prefix_missing_trigger=AUTO_PREFIX_MISSING_TRIGGER,
    minimum_pair_count=MINIMUM_PAIR_COUNT,
    expected_pair_count=EXPECTED_PAIR_COUNT,
    fail_on_exact_duplicates=FAIL_ON_EXACT_DUPLICATES,
    near_duplicate_hamming_threshold=NEAR_DUPLICATE_HAMMING_THRESHOLD,
    accepted_image_extensions=ACCEPTED_IMAGE_EXTENSIONS,
    gallery_columns=GALLERY_COLUMNS,
    gallery_thumbnail_size=GALLERY_THUMBNAIL_SIZE,
    gallery_page_size=GALLERY_PAGE_SIZE,
)

dataset = pipeline.prepare_dataset(
    zip_path=DATASET_ZIP,
    config=dataset_config,
)

dataset.display_summary()
dataset.show_gallery(
    columns=GALLERY_COLUMNS,
    thumbnail_size=GALLERY_THUMBNAIL_SIZE,
    page_size=GALLERY_PAGE_SIZE,
    show_filename=True,
    show_dimensions=True,
    show_caption=True,
    highlight_trigger=True,
)
dataset.show_caption_audit()
dataset.show_issues()
