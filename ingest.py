#!/usr/bin/env python3
"""
wiki/ingest.py — Ingest large PDFs into the wiki with minimum information loss.

Usage:
    python3 ingest.py <path-to-pdf> [--out <slug>] [--chunk-pages 30]

What it does:
    1. Extracts all text from the PDF using pdfplumber (no page limit).
    2. Splits into chunks (default 30 pages) to stay within Claude's context.
    3. Calls Claude for each chunk to extract key facts, concepts, and entities.
    4. Calls Claude a final time to merge chunk summaries into wiki pages:
         sources/<slug>.md        — full source summary
         concepts/<slug>-*.md     — any new concept pages
         entities/<slug>-*.md     — any new entity pages
    5. Updates index.md and appends to log.md.

Requirements: pip install pypdf pdfplumber anthropic pytesseract pdf2image
              brew install tesseract tesseract-lang poppler
Environment:  ANTHROPIC_API_KEY must be set.
"""

import argparse
import os
import re
import sys
import textwrap
from datetime import date
from pathlib import Path

import pdfplumber
import anthropic
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # allow very large scanned pages

WIKI_ROOT = Path(__file__).parent
SCHEMA = (WIKI_ROOT / "schema.md").read_text()
TODAY = date.today().isoformat()

# Load API key from env or ~/.anthropic_key (never committed to git)
if not os.environ.get("ANTHROPIC_API_KEY"):
    key_file = Path.home() / ".anthropic_key"
    if key_file.exists():
        os.environ["ANTHROPIC_API_KEY"] = key_file.read_text().strip()
    else:
        sys.exit("ANTHROPIC_API_KEY environment variable is not set.")

CLIENT = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_text(pdf_path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for every page in the PDF.
    Falls back to OCR (Tesseract chi_sim+eng) for image-only pages."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        native_pages = list(pdf.pages)
        total_native_chars = sum(len(p.extract_text() or "") for p in native_pages)

    needs_ocr = total_native_chars == 0
    if needs_ocr:
        print("[ingest] No embedded text detected — falling back to OCR (chi_sim+eng) …")
        try:
            from pdf2image import convert_from_path
            import pytesseract
        except ImportError:
            sys.exit("OCR fallback requires: pip install pdf2image pytesseract  &&  brew install poppler tesseract-lang")

        images = convert_from_path(pdf_path, dpi=150)
        for i, img in enumerate(images, start=1):
            print(f"[ingest] OCR page {i}/{len(images)} …")
            text = pytesseract.image_to_string(img, lang="chi_sim+eng")
            pages.append((i, text))
    else:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append((i, text))
    return pages


def chunk_pages(pages: list[tuple[int, str]], chunk_size: int) -> list[list[tuple[int, str]]]:
    """Split page list into chunks of at most chunk_size pages."""
    return [pages[i : i + chunk_size] for i in range(0, len(pages), chunk_size)]


def call_claude(prompt: str, max_tokens: int = 4096) -> str:
    msg = CLIENT.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CHUNK_PROMPT = textwrap.dedent("""\
    You are helping build a personal wiki. Below is a slice of a document (pages {start}–{end}).
    Extract ALL meaningful information with minimum loss:
    - Key facts, arguments, claims, data points
    - Named concepts (with brief definitions)
    - Named entities: people, organizations, tools, products
    - Any terminology specific to this domain

    Be thorough. Do not summarise away details. Use bullet points grouped by topic.
    Preserve numbers, dates, citations, and technical terms verbatim.
    Write all output in English — translate any Chinese or other non-English text.

    === DOCUMENT SLICE (pages {start}–{end}) ===
    {text}
""")

MERGE_PROMPT = textwrap.dedent("""\
    You are maintaining a personal wiki that follows this schema:
    {schema}

    The document being ingested is: "{title}"
    File path in sources/: {filename}
    Today's date: {today}

    Below are structured notes extracted chunk-by-chunk from the full document.
    Write ALL wiki output in English — translate any Chinese or other non-English text in the source.
    Using ALL of these notes, produce exactly the following wiki output — nothing else:

    ### FILE: sources/{slug}.md
    (Full source summary page following schema.md conventions — include all important details,
     do not omit anything significant. Must be thorough, not a brief abstract.)

    ### FILE: concepts/<slug>-<concept-slug>.md
    (One file per significant concept introduced in this document. Omit if there are none.)

    ### FILE: entities/<slug>-<entity-slug>.md
    (One file per significant entity (person, org, tool, product). Omit if there are none.)

    ### INDEX_ENTRY
    (Single line to add to index.md under the appropriate category)

    ### LOG_ENTRY
    (Single line: ## [{today}] ingest | {title})

    Use the exact "### FILE: path/name.md" header format so the output can be parsed.

    === CHUNK NOTES ===
    {notes}
""")


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_output(text: str) -> dict[str, str]:
    """Parse ### FILE: path blocks and special ### INDEX_ENTRY / ### LOG_ENTRY."""
    result: dict[str, str] = {}
    # Split on any ### header
    parts = re.split(r"(###\s+\S[^\n]*)", text)
    current_key = None
    for part in parts:
        if part.startswith("###"):
            header = part[3:].strip()
            if header.startswith("FILE:"):
                current_key = header[5:].strip()
            elif header in ("INDEX_ENTRY", "LOG_ENTRY"):
                current_key = header
            else:
                current_key = None
        elif current_key is not None:
            result[current_key] = part.strip()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def ingest(pdf_path: Path, slug: str, chunk_size: int) -> None:
    print(f"[ingest] Reading {pdf_path.name} …")
    pages = extract_text(pdf_path)
    total = len(pages)
    print(f"[ingest] {total} pages extracted")

    chunks = chunk_pages(pages, chunk_size)
    chunk_notes: list[str] = []

    for idx, chunk in enumerate(chunks):
        start_page = chunk[0][0]
        end_page = chunk[-1][0]
        print(f"[ingest] Processing chunk {idx+1}/{len(chunks)} (pages {start_page}–{end_page}) …")
        text = "\n\n".join(t for _, t in chunk)
        prompt = CHUNK_PROMPT.format(start=start_page, end=end_page, text=text)
        notes = call_claude(prompt, max_tokens=8192)
        chunk_notes.append(f"[Pages {start_page}–{end_page}]\n{notes}")

    print("[ingest] Merging chunks into wiki pages …")
    all_notes = "\n\n---\n\n".join(chunk_notes)
    merge_prompt = MERGE_PROMPT.format(
        schema=SCHEMA,
        title=pdf_path.stem,
        filename=pdf_path.name,
        slug=slug,
        today=TODAY,
        notes=all_notes,
    )
    merged = call_claude(merge_prompt, max_tokens=16000)

    files = parse_output(merged)

    # Write files
    for path_str, content in files.items():
        if path_str in ("INDEX_ENTRY", "LOG_ENTRY"):
            continue
        out_path = WIKI_ROOT / path_str
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content + "\n")
        print(f"[ingest] Wrote {path_str}")

    # Update index.md
    index_entry = files.get("INDEX_ENTRY", "")
    if index_entry:
        index_path = WIKI_ROOT / "index.md"
        with open(index_path, "a") as f:
            f.write(f"\n{index_entry}\n")
        print("[ingest] Updated index.md")

    # Append to log.md
    log_entry = files.get("LOG_ENTRY", f"## [{TODAY}] ingest | {pdf_path.stem}")
    log_path = WIKI_ROOT / "log.md"
    with open(log_path, "a") as f:
        f.write(f"\n{log_entry}\n")
    print("[ingest] Appended to log.md")

    print("[ingest] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a large PDF into the wiki.")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--out", help="Output slug (default: sanitized PDF filename)")
    parser.add_argument("--chunk-pages", type=int, default=30,
                        help="Pages per chunk sent to Claude (default: 30)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        sys.exit(f"File not found: {pdf_path}")

    slug = args.out or re.sub(r"[^a-z0-9]+", "-", pdf_path.stem.lower()).strip("-")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY environment variable is not set.")

    ingest(pdf_path, slug, args.chunk_pages)


if __name__ == "__main__":
    main()
