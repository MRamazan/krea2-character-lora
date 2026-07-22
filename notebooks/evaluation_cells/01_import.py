# ============================================================
# Upload and import evaluation bundle
# ============================================================

from google.colab import files

uploaded = files.upload()
zip_files = [name for name in uploaded if name.lower().endswith(".zip")]

if len(zip_files) != 1:
    raise RuntimeError("Upload exactly one evaluation bundle ZIP file.")

EVALUATION_BUNDLE_ZIP = f"/content/{zip_files[0]}"

imported_run = pipeline.import_evaluation_bundle(
    zip_path=EVALUATION_BUNDLE_ZIP,
)

imported_run.display_summary()
