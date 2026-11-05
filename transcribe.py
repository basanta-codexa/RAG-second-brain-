"""Transcribe a PDF of handwritten notes with a vision-capable Gemini model.

Pages are cached one file each under --cache-dir, so reruns resume where they
stopped. A page is only cached when the model returns non-empty text, so a
failed or empty page is always retried instead of silently skipped.

Reads the API key from the GEMINI_API_KEY environment variable.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import pymupdf as fitz  # PyMuPDF
from google import genai
from google.genai import types

DEFAULT_PDF = "system-design-notes.pdf"
DEFAULT_OUT = "data/system-design-notes.md"
DEFAULT_CACHE_DIR = "transcripts"
DEFAULT_MODEL = "gemini-3.8-flash"  # change if Google retires it
DEFAULT_ZOOM = 2.0                 # 2x resolution keeps handwriting readable
DEFAULT_PAUSE = 7.0                # stay under free-tier requests-per-minute
DEFAULT_MAX_RETRIES = 4

PROMPT = """Transcribe this page of handwritten study notes into clean Markdown.
Rules:
- Be faithful. Do not add, explain, or fix anything the page does not say.
- Keep the reading order a person would follow (title, then boxes, then arrows and side notes).
- Write labels connected by arrows as "A -> B".
- For a diagram, write one line: [Diagram: short description], then transcribe its labels.
- Output only the transcription, with no introduction or commentary."""


def cache_file(cache_dir: Path, page_number: int) -> Path:
    return cache_dir / f"page_{page_number:03d}.md"


def cached_pages(cache_dir: Path, total: int) -> tuple[dict[int, str], list[int]]:
    """Return (non-empty cached pages, missing/empty page numbers)."""
    done: dict[int, str] = {}
    missing: list[int] = []
    for i in range(1, total + 1):
        p = cache_file(cache_dir, i)
        if p.exists() and p.stat().st_size > 0:
            text = p.read_text(encoding="utf-8").strip()
            if text:
                done[i] = text
                continue
        missing.append(i)
    return done, missing


def transcribe_page(client, model: str, png: bytes, max_retries: int) -> str:
    """Call the model once, retrying transient failures; raise on permanent failure."""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=png, mime_type="image/png"),
                    PROMPT,
                ],
            )
            text = (resp.text or "").strip()
            if not text:
                raise ValueError("model returned an empty transcription")
            return text
        except Exception as exc:  # API/network/empty-response failures
            last_error = exc
            backoff = min(120.0, 20.0 * (2 ** (attempt - 1)))  # 20, 40, 80, 120 ...
            wait = backoff * (0.75 + 0.5 * random.random())    # jittered
            print(
                f"    attempt {attempt}/{max_retries} failed: "
                f"{type(exc).__name__}: {str(exc)[:140]} -> retry in {wait:.0f}s"
            )
            if attempt < max_retries:
                time.sleep(wait)
    raise RuntimeError(f"page failed after {max_retries} attempts: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe a handwritten-notes PDF to Markdown.")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help=f"input PDF (default: {DEFAULT_PDF})")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"output markdown (default: {DEFAULT_OUT})")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help=f"per-page cache directory (default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL})")
    parser.add_argument("--zoom", type=float, default=DEFAULT_ZOOM, help=f"render zoom (default: {DEFAULT_ZOOM})")
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE, help=f"seconds between calls (default: {DEFAULT_PAUSE})")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help=f"per-page retries (default: {DEFAULT_MAX_RETRIES})")
    parser.add_argument("--force", action="store_true", help="re-transcribe every page, ignoring the cache")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY is not set in the environment.", file=sys.stderr)
        return 2

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    total = len(doc)
    print(f"{total} pages to transcribe with {args.model}")

    client = genai.Client()  # reads GEMINI_API_KEY from the environment

    done, missing = cached_pages(cache_dir, total)
    if args.force:
        missing = list(range(1, total + 1))

    print(f"cache: {len(done)} pages done, {len(missing)} to fetch")
    failed: list[int] = []

    for page_number in missing:
        page = doc.load_page(page_number - 1)
        png = page.get_pixmap(matrix=fitz.Matrix(args.zoom, args.zoom)).tobytes("png")
        print(f"page {page_number}: transcribing...")
        try:
            text = transcribe_page(client, args.model, png, args.max_retries)
        except RuntimeError as exc:
            print(f"page {page_number}: FAILED ({exc})", file=sys.stderr)
            failed.append(page_number)
            continue
        cache_file(cache_dir, page_number).write_text(text, encoding="utf-8")
        print(f"page {page_number}: ok ({len(text)} chars)")
        time.sleep(args.pause)

    if failed:
        print(
            f"\n{len(failed)} page(s) failed: {', '.join(map(str, failed))}. "
            "Rerun this command to resume; nothing was lost.",
            file=sys.stderr,
        )
        return 1

    # Atomically write the combined markdown; require ALL pages to be cached.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    parts = []
    for i in range(1, total + 1):
        body = cache_file(cache_dir, i).read_text(encoding="utf-8").strip()
        if not body:  # paranoia: never emit a blank page silently
            print(f"ERROR: page {i} is empty in cache; aborting write.", file=sys.stderr)
            return 1
        parts.append(f"<<<PAGE {i}>>>\n{body}")
    tmp_path.write_text("\n\n".join(parts), encoding="utf-8")
    tmp_path.replace(out_path)  # atomic on the same filesystem
    print(f"\nWrote {out_path} ({total} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
