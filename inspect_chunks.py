"""Inspect what the chunker produced: find passages or dump a specific page.

Usage:
    python inspect_chunks.py find "cap theorem"
    python inspect_chunks.py page system-design-notes.md 12
"""
import argparse
import sys

from chunker import chunk_records
from loader import load_folder


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect generated chunks.")
    parser.add_argument("--folder", default="data", help="corpus folder (default: data)")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    sub = parser.add_subparsers(dest="mode", required=True)

    find_p = sub.add_parser("find", help="find chunks containing a case-insensitive term")
    find_p.add_argument("term")

    page_p = sub.add_parser("page", help="dump all chunks from one page of one source")
    page_p.add_argument("source")
    page_p.add_argument("page", type=int)
    args = parser.parse_args()

    chunks = chunk_records(load_folder(args.folder), args.chunk_size, args.overlap)

    if args.mode == "find":
        term = args.term.lower()
        found = [c for c in chunks if term in c["text"].lower()]
        print(f"{len(found)} chunks contain '{term}'\n")
        for c in found[:5]:
            print(f"[{c['source']} | p{c['page']}]")
            print(c["text"][:400], "\n" + "-" * 40)
        return 0

    for c in chunks:
        if c["source"] == args.source and c["page"] == args.page:
            print(c["text"], "\n" + "-" * 40)
    return 0


if __name__ == "__main__":
    sys.exit(main())
