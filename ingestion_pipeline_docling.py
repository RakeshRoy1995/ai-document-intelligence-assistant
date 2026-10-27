"""
Step 1 of the pipeline: PDF -> flat list of layout items (JSON).

    .venv\\Scripts\\python.exe ingestion_pipeline_docling.py     # every PDF in docs/

Docling detects the reading order and the type of every block on the page.
Each block becomes one item:

    {
      "id": "report.pdf#0012",
      "source": "report.pdf",
      "type": "heading" | "text" | "list_item" | "table" | "figure" | "equation" | "code" | "caption",
      "text": "...",            # tables as Markdown, figures as their caption
      "page": 3,
      "level": 2,               # headings only
      "image": "report_p3_fig1.png"   # figures only, file in IMAGE_DIR
    }
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

DOCS_PATH = os.environ.get("DOCS_PATH", "docs")
DATA_DIR = os.environ.get("DATA_DIR", "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
OUTPUT_JSON = os.path.join(DATA_DIR, "layout_items.json")

# OCR is only needed for scanned PDFs and makes extraction much slower
DO_OCR = os.environ.get("DOCLING_OCR", "0") == "1"
IMAGE_SCALE = 2.0

# Docling labels -> our item types. Labels not listed here are dropped.
LABEL_TO_TYPE = {
    "title": "heading",
    "section_header": "heading",
    "text": "text",
    "paragraph": "text",
    "footnote": "text",
    "reference": "text",
    "list_item": "list_item",
    "caption": "caption",
    "table": "table",
    "document_index": "table",
    "picture": "figure",
    "chart": "figure",
    "formula": "equation",
    "code": "code",
}
# page_header, page_footer, etc. are layout noise for RAG


# ============================================================
# CONVERTER
# ============================================================

def build_converter():
    """Docling models are big; build this once and reuse it."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = DO_OCR
    options.do_table_structure = True
    options.generate_picture_images = True
    options.images_scale = IMAGE_SCALE

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


# ============================================================
# DOCLING DOCUMENT -> ITEMS
# ============================================================

def _page(item) -> int:
    prov = getattr(item, "prov", None)
    return prov[0].page_no if prov else 0


def _clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", (text or "")).strip()


def _safe_stem(source: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(source).stem)[:60]


def convert_document(doc, source: str, image_dir) -> List[Dict[str, Any]]:
    """Walk a DoclingDocument in reading order and flatten it into items."""
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Remove figures from an earlier upload of the same file
    stem = _safe_stem(source)
    for old in image_dir.glob(f"{stem}_p*_fig*.png"):
        old.unlink(missing_ok=True)

    items: List[Dict[str, Any]] = []
    figure_count = 0

    for node, _depth in doc.iterate_items():
        label = getattr(getattr(node, "label", None), "value", str(getattr(node, "label", "")))
        item_type = LABEL_TO_TYPE.get(label)
        if item_type is None:
            continue

        page = _page(node)
        record: Dict[str, Any] = {"source": source, "type": item_type, "page": page}

        if item_type == "table":
            try:
                text = node.export_to_markdown(doc=doc)
            except TypeError:
                text = node.export_to_markdown()
            caption = _clean(node.caption_text(doc)) if hasattr(node, "caption_text") else ""
            if caption:
                record["caption"] = caption
            record["text"] = text.strip()

        elif item_type == "figure":
            figure_count += 1
            caption = _clean(node.caption_text(doc)) if hasattr(node, "caption_text") else ""
            record["text"] = caption
            try:
                image = node.get_image(doc)
            except Exception:
                image = None
            if image is not None:
                name = f"{stem}_p{page}_fig{figure_count}.png"
                image.save(image_dir / name)
                record["image"] = name

        else:
            record["text"] = _clean(getattr(node, "text", ""))
            if item_type == "heading":
                # title = level 0, section headers start at 1
                record["level"] = 0 if label == "title" else int(getattr(node, "level", 1) or 1)

        if not record.get("text") and not record.get("image"):
            continue
        items.append(record)

    for i, record in enumerate(items):
        record["id"] = f"{source}#{i:04d}"
    return items


# ============================================================
# CLI
# ============================================================

def main():
    pdfs = sorted(Path(DOCS_PATH).glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {DOCS_PATH}/")
        return

    converter = build_converter()
    all_items: List[Dict[str, Any]] = []
    for pdf in pdfs:
        print(f"Extracting {pdf.name} ...")
        result = converter.convert(pdf)
        items = convert_document(result.document, pdf.name, IMAGE_DIR)
        print(f"  {len(items)} items")
        all_items.extend(items)

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(all_items)} items -> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
