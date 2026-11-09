"""Split page-level records into overlapping, sentence-aware chunks."""
from __future__ import annotations

import argparse
import re

from loader import load_folder

_SENTENCE_END = re.compile(r"""[.!?]["')\]]?(?=\s)""")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks, preferring to cut at sentence ends."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = " ".join(text.split())  # collapse messy whitespace/newlines
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)

        # try to end at a sentence boundary instead of mid-sentence
        if end < n:
            cut = None
            for match in _SENTENCE_END.finditer(text, start, end):
                if match.end() > start + chunk_size // 2:
                    cut = match.end()
            if cut is not None:
                end = cut  # keep the punctuation with this chunk

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_records(records: list[dict], chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    """Turn page-level records into chunk-level records, keeping metadata."""
    all_chunks: list[dict] = []
    for rec in records:
        pieces = chunk_text(rec["text"], chunk_size, overlap)
        origin = rec.get("origin") or rec["source"]
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "text": piece,
                "source": rec["source"],
                "page": rec["page"],
                "chunk_id": f"{origin}-p{rec['page']}-c{i}",
                "origin": origin,
            })
    return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview chunking for a folder.")
    parser.add_argument("--folder", default="data", help="folder to load (default: data)")
    parser.add_argument("--chunk-size", type=int, default=800, help="chunk size in characters")
    parser.add_argument("--overlap", type=int, default=150, help="overlap in characters")
    parser.add_argument("--show", type=int, default=3, help="number of sample chunks to print")
    args = parser.parse_args()

    records = load_folder(args.folder)
    chunks = chunk_records(records, args.chunk_size, args.overlap)
    print(f"{len(records)} pages -> {len(chunks)} chunks")
    if not chunks:
        print("No chunks produced (nothing with readable text in the folder).")
        return

    lengths = [len(c["text"]) for c in chunks]
    print(f"Chunk length: min {min(lengths)}, max {max(lengths)}, avg {sum(lengths) // len(lengths)}")
    print()
    for c in chunks[: args.show]:
        print(f"[{c['chunk_id']}]")
        print(c["text"])
        print("-" * 40)


if __name__ == "__main__":
    main()
