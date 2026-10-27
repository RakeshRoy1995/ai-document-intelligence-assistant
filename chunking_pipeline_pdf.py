"""
Step 2 of the pipeline: layout items -> section-aware RAG chunks (JSON).

    .venv\\Scripts\\python.exe chunking_pipeline_pdf.py     # data/layout_items.json -> data/chunks.json

Headings open sections. Inside a section, text / lists / equations / code are
packed into text chunks of about CHUNK_SIZE characters with a little overlap.
Tables and figures always get their own chunk so they can be filtered and
shown on their own. Every chunk carries its section path, so the embedding
knows where in the document the text comes from.

Chunk record:

    {
      "chunk_id": "a3f9c1...",        # stable hash of source + section + content
      "chunk_index": 17,              # unique across all documents
      "source": "report.pdf",
      "chunk_type": "text" | "table" | "figure",
      "section_path": "2 Methods > 2.1 Data",
      "content": "...",
      "page_start": 3, "page_end": 4,
      "item_ids": ["report.pdf#0031", ...],
      "image": "report_p4_fig2.png",  # figures only
      "char_count": 912
    }
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR", "data")
INPUT_JSON = os.path.join(DATA_DIR, "layout_items.json")
OUTPUT_JSON = os.path.join(DATA_DIR, "chunks.json")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1200"))        # characters
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
MAX_TABLE_CHARS = 4000                                         # larger tables are split by rows

TEXT_TYPES = {"text", "list_item", "equation", "code", "caption"}


# ============================================================
# SECTIONS
# ============================================================

def build_sections(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group items under their nearest heading, tracking the heading path."""
    sections: List[Dict[str, Any]] = []
    stack: List[tuple] = []          # (level, heading text)
    current = None

    def open_section(source: str, page: int):
        nonlocal current
        current = {
            "source": source,
            "heading": stack[-1][1] if stack else "",
            "section_path": " > ".join(h for _, h in stack),
            "level": stack[-1][0] if stack else 0,
            "page_start": page,
            "page_end": page,
            "items": [],
        }
        sections.append(current)

    for item in items:
        source = item["source"]
        if current is not None and current["source"] != source:
            stack, current = [], None

        if item["type"] == "heading":
            level = item.get("level", 1)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, item["text"]))
            open_section(source, item["page"])
            continue

        if current is None:
            open_section(source, item["page"])
        current["items"].append(item)
        current["page_end"] = max(current["page_end"], item["page"])

    return [s for s in sections if s["items"]]


# ============================================================
# CHUNKS
# ============================================================

def _chunk_id(source: str, section_path: str, content: str) -> str:
    return hashlib.sha1(f"{source}\x00{section_path}\x00{content}".encode("utf-8")).hexdigest()[:20]


def _item_text(item: Dict[str, Any]) -> str:
    text = item.get("text", "")
    if item["type"] == "list_item":
        return f"- {text}"
    if item["type"] == "equation":
        return f"$$ {text} $$"
    if item["type"] == "code":
        return f"```\n{text}\n```"
    return text


def _split_long(text: str, size: int, overlap: int) -> List[str]:
    """Split one oversized paragraph on sentence / word boundaries."""
    parts, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return parts


def _pack_text(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Greedy packing of paragraphs into ~CHUNK_SIZE groups, overlapping by tail paragraphs."""
    units = []   # (text, page, item_id)
    for it in items:
        text = _item_text(it)
        if len(text) > CHUNK_SIZE:
            units += [(p, it["page"], it["id"]) for p in _split_long(text, CHUNK_SIZE, CHUNK_OVERLAP)]
        else:
            units.append((text, it["page"], it["id"]))

    groups, current, length = [], [], 0
    for unit in units:
        if current and length + len(unit[0]) > CHUNK_SIZE:
            groups.append(current)
            # carry trailing paragraphs forward as overlap (never the whole group)
            tail, tail_len = [], 0
            for u in reversed(current[1:]):
                if tail_len + len(u[0]) > CHUNK_OVERLAP:
                    break
                tail.insert(0, u)
                tail_len += len(u[0])
            current, length = tail, tail_len
        current.append(unit)
        length += len(unit[0])
    if current:
        groups.append(current)

    return [
        {
            "content": "\n\n".join(u[0] for u in g),
            "page_start": min(u[1] for u in g),
            "page_end": max(u[1] for u in g),
            "item_ids": list(dict.fromkeys(u[2] for u in g)),
        }
        for g in groups
    ]


def _split_table(markdown: str) -> List[str]:
    """Split a big Markdown table by rows, repeating the header in each part."""
    if len(markdown) <= MAX_TABLE_CHARS:
        return [markdown]
    lines = markdown.splitlines()
    header, rows = lines[:2], lines[2:]
    parts, current = [], []
    for row in rows:
        if current and len("\n".join(header + current + [row])) > MAX_TABLE_CHARS:
            parts.append("\n".join(header + current))
            current = []
        current.append(row)
    if current:
        parts.append("\n".join(header + current))
    return parts


def create_chunks(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []

    def add(section, chunk_type, content, page_start, page_end, item_ids, **extra):
        content = content.strip()
        if not content:
            return
        chunks.append({
            "chunk_id": _chunk_id(section["source"], section["section_path"], f"{chunk_type}:{content}"),
            "chunk_index": len(chunks),
            "source": section["source"],
            "chunk_type": chunk_type,
            "section_path": section["section_path"],
            "content": content,
            "page_start": page_start,
            "page_end": page_end,
            "item_ids": item_ids,
            **extra,
            "char_count": len(content),
        })

    for section in sections:
        pending_text: List[Dict[str, Any]] = []

        def flush():
            for group in _pack_text(pending_text):
                add(section, "text", group["content"], group["page_start"], group["page_end"], group["item_ids"])
            pending_text.clear()

        for item in section["items"]:
            if item["type"] in TEXT_TYPES:
                pending_text.append(item)

            elif item["type"] == "table":
                flush()
                caption = item.get("caption", "")
                for part in _split_table(item["text"]):
                    content = f"Table: {caption}\n\n{part}" if caption else part
                    add(section, "table", content, item["page"], item["page"], [item["id"]])

            elif item["type"] == "figure":
                flush()
                caption = item.get("text") or "Figure (no caption)"
                extra = {"image": item["image"]} if item.get("image") else {}
                add(section, "figure", f"Figure: {caption}", item["page"], item["page"], [item["id"]], **extra)

        flush()

    # Identical content in the same section (e.g. repeated boilerplate) -> keep first
    seen, unique = set(), []
    for c in chunks:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            unique.append(c)
    for i, c in enumerate(unique):
        c["chunk_index"] = i
    return unique


# ============================================================
# CLI
# ============================================================

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        items = json.load(f)
    sections = build_sections(items)
    chunks = create_chunks(sections)
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"{len(items)} items -> {len(sections)} sections -> {len(chunks)} chunks -> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
