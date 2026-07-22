import json
import math
import shutil
import textwrap
from pathlib import Path
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont

validation = json.loads((PATHS["dataset_audit"] / "dataset_validation_manifest.json").read_text(encoding="utf-8"))
inspection_directory = PATHS["dataset_audit"] / "inspection_pages"
if inspection_directory.exists():
    shutil.rmtree(inspection_directory)
inspection_directory.mkdir(parents=True, exist_ok=False)

font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
font = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else ImageFont.load_default()
small_font = ImageFont.truetype(str(font_path), 15) if font_path.is_file() else ImageFont.load_default()
columns = 3
rows = 2
items_per_page = columns * rows
cell_width = 520
cell_height = 670
image_box = 460
page_records = []

for page_index in range(math.ceil(len(validation["records"]) / items_per_page)):
    page_items = validation["records"][page_index * items_per_page:(page_index + 1) * items_per_page]
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for local_index, item in enumerate(page_items):
        column = local_index % columns
        row = local_index // columns
        x = column * cell_width
        y = row * cell_height
        image = Image.open(item["image"]).convert("RGB")
        image.thumbnail((image_box, image_box), Image.Resampling.LANCZOS)
        image_x = x + (cell_width - image.width) // 2
        image_y = y + 12
        canvas.paste(image, (image_x, image_y))
        information_y = y + image_box + 24
        draw.text((x + 18, information_y), f"{Path(item['image']).name} | {item['source_key']}", fill="black", font=font)
        draw.text((x + 18, information_y + 27), f"{item['width']} x {item['height']} | ratio {item['aspect_ratio']:.3f}", fill="black", font=small_font)
        caption_lines = textwrap.wrap(item["text"], width=58)[:6]
        for line_index, line in enumerate(caption_lines):
            draw.text((x + 18, information_y + 52 + line_index * 19), line, fill="black", font=small_font)
    page_path = inspection_directory / f"inspection_page_{page_index + 1:03d}.png"
    canvas.save(page_path)
    page_records.append(str(page_path))
    display(canvas)

manifest = {"page_count": len(page_records), "items_per_page": items_per_page, "pages": page_records}
manifest_path = PATHS["dataset_audit"] / "inspection_pages_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Inspection pages: {manifest_path}")