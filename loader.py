"""Load source documents into page-level records.

A record is a dict with:
    text   -- full page/file text
    source -- file name (used for citations)
    page   -- 1-based page number, or 0 for unmarked front matter
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from pypdf import PdfReader

_PAGE_MARKER = re.compile(r"<<<PAGE (\d+)>>>")
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def load_pdf(path: Path) -> list[dict]:
    """Read a PDF and return one record per non-empty page."""
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")  # only password-less encryption can continue
            except Exception as exc:
                raise RuntimeError(f"encrypted PDF cannot be opened: {path}") from exc
        records = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                records.append({"text": text, "source": path.name, "page": page_number})
        return records
    except Exception as exc:
        raise RuntimeError(f"failed to read PDF {path}: {exc}") from exc


def load_text(path: Path) -> list[dict]:
    """Read a .txt/.md file, splitting on <<<PAGE n>>> markers when present.

    Content before the first marker is kept as page 0 instead of being dropped.
    """
    text = path.read_text(encoding="utf-8")
    parts = _PAGE_MARKER.split(text)
    if len(parts) == 1:
        return [{"text": text, "source": path.name, "page": 1}]

    records: list[dict] = []
    front = parts[0].strip()
    if front:
        records.append({"text": parts[0], "source": path.name, "page": 0})
    # parts looks like [front, "1", text1, "2", text2, ...]
    for num, body in zip(parts[1::2], parts[2::2]):
        if body.strip():
            records.append({"text": body, "source": path.name, "page": int(num)})
    return records


def load_folder(folder: str, strict: bool = True) -> list[dict]:
    """Load every supported file in *folder* (non-recursive).

    With strict=True (default) an unreadable file aborts the load instead of
    silently dropping content. Set strict=False to log-and-continue.
    """
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"data folder not found: {root}")

    records: list[dict] = []
    for path in sorted(root.iterdir()):
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                records.extend(load_pdf(path))
            elif suffix in (".txt", ".md"):
                records.extend(load_text(path))
            else:
                print(f"Skipping unsupported file: {path.name}")
        except RuntimeError as exc:
            if strict:
                raise
            print(f"ERROR: {exc} -- skipping file")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and list source records.")
    parser.add_argument("folder", nargs="?", default="data", help="folder to load (default: data)")
    args = parser.parse_args()

    docs = load_folder(args.folder)
    print(f"Loaded {len(docs)} pieces\n")
    for d in docs[:3]:
        print(f"[{d['source']} | page {d['page']}]")
        print(d["text"][:200].replace("\n", " "))
        print("-" * 40)


if __name__ == "__main__":
    main()
