import os
import sys
import time
from pathlib import Path

import pymupdf as fitz  # PyMuPDF
from google import genai
from google.genai import types

# ---- settings ----
# the original PDF (keep it OUTSIDE the data folder)
PDF_PATH = "system-design-notes.pdf"
OUT_PATH = "data/system-design-notes.md"
CACHE_DIR = Path("transcripts")          # one file per page, so reruns resume
MODEL = "gemini-3.8-flash"               # change if Google has retired it
ZOOM = 2.0                               # 2x resolution so handwriting stays readable
# stay under the free-tier requests-per-minute limit
PAUSE_SECONDS = 7
# ------------------

PROMPT = """Transcribe this page of handwritten study notes into clean Markdown.
Rules:
- Be faithful. Do not add, explain, or fix anything the page does not say.
- Keep the reading order a person would follow (title, then boxes, then arrows and side notes).
- Write labels connected by arrows as "A -> B".
- For a diagram, write one line: [Diagram: short description], then transcribe its labels.
- Output only the transcription, with no introduction or commentary."""

  # reads GEMINI_API_KEY from the environment

CACHE_DIR.mkdir(exist_ok=True)
doc = fitz.open(PDF_PATH)
print(f"{len(doc)} pages to transcribe with {MODEL}")

for i, page in enumerate(doc, start=1):
    cache_file = CACHE_DIR / f"page_{i:03d}.md"
    if cache_file.exists():
        print(f"page {i}: already done, skipping")
        continue
    client = genai.Client()
    png = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM)).tobytes("png")

    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=[types.Part.from_bytes(
                    data=png, mime_type="image/png"), PROMPT],
            )
            text = (resp.text or "").strip()
            cache_file.write_text(text, encoding="utf-8")
            print(f"page {i}: ok ({len(text)} chars)")
            break
        except Exception as e:
            wait = 20 * (attempt + 1)
            print(
                f"page {i}: error {type(e).__name__}: {str(e)[:120]} -> retry in {wait}s")
            time.sleep(wait)
    else:
        print(
            f"page {i}: FAILED after retries, rerun the script later to resume")
        sys.exit(1)

    time.sleep(PAUSE_SECONDS)

# Combine all pages, with a marker line so our loader can recover page numbers
parts = []
for i in range(1, len(doc) + 1):
    f = CACHE_DIR / f"page_{i:03d}.md"
    parts.append(f"<<<PAGE {i}>>>\n{f.read_text(encoding='utf-8')}")
Path(OUT_PATH).write_text("\n\n".join(parts), encoding="utf-8")
print(f"\nWrote {OUT_PATH}")
