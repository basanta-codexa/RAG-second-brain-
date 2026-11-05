"""Cosine-similarity search over the corpus, with a persistent embedding cache.

Usage:
    python search.py                    # interactive REPL
    python search.py --question "..."   # one-shot answer
    python search.py --rebuild          # re-embed everything, ignore cache

Embeddings are cached under .cache/ keyed by a content hash of the corpus and
the model/chunk settings, so unchanged corpora load instantly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from loader import load_folder, SUPPORTED_SUFFIXES
from chunker import chunk_records

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_CHUNK_SIZE = 800      # BGE-en reads ~512 tokens; keep chunks modest
DEFAULT_OVERLAP = 150
DEFAULT_TOP_K = 3
CACHE_DIR = Path(".cache")
INDEX_PATH = CACHE_DIR / "index.npz"
META_PATH = CACHE_DIR / "index_meta.json"

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def corpus_fingerprint(folder: str) -> str:
    """SHA-256 over file names and contents of every supported file."""
    root = Path(folder)
    h = hashlib.sha256()
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        h.update(path.name.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def cache_is_fresh(fingerprint: str, model: str, chunk_size: int, overlap: int) -> bool:
    if not INDEX_PATH.exists() or not META_PATH.exists():
        return False
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        meta.get("fingerprint") == fingerprint
        and meta.get("model") == model
        and meta.get("chunk_size") == chunk_size
        and meta.get("overlap") == overlap
    )


def save_cache(chunks: list[dict], vectors: np.ndarray, fingerprint: str,
               model: str, chunk_size: int, overlap: int) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        INDEX_PATH,
        vectors=vectors.astype(np.float32),
        texts=np.array([c["text"] for c in chunks]),
        sources=np.array([c["source"] for c in chunks]),
        pages=np.array([c["page"] for c in chunks], dtype=np.int64),
        chunk_ids=np.array([c["chunk_id"] for c in chunks]),
    )
    META_PATH.write_text(json.dumps({
        "fingerprint": fingerprint,
        "model": model,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunks": len(chunks),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2), encoding="utf-8")


def load_cache() -> tuple[list[dict], np.ndarray]:
    blob = np.load(INDEX_PATH, allow_pickle=False)
    chunks = [
        {"text": t, "source": s, "page": int(p), "chunk_id": cid}
        for t, s, p, cid in zip(
            blob["texts"], blob["sources"], blob["pages"], blob["chunk_ids"]
        )
    ]
    return chunks, blob["vectors"]


def create_index(folder: str = "data", model_name: str = DEFAULT_MODEL,
                 chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP,
                 rebuild: bool = False) -> dict:
    """Load the corpus, embed it (or reuse the cache), return a ready index dict."""
    fingerprint = corpus_fingerprint(folder)

    if not rebuild and cache_is_fresh(fingerprint, model_name, chunk_size, overlap):
        chunks, vectors = load_cache()
        print(f"cache hit: reused {len(chunks)} embeddings from {INDEX_PATH}")
        return {"model_name": model_name, "chunks": chunks, "vectors": vectors,
                "fingerprint": fingerprint, "encoder": None}

    print(f"loading files and chunking {folder}...")
    records = load_folder(folder)
    chunks = chunk_records(records, chunk_size, overlap)
    if not chunks:
        raise RuntimeError(f"no chunks produced from {folder}; is it empty?")
    print(f"{len(chunks)} chunks; embedding with {model_name} ...")
    model = SentenceTransformer(model_name)
    t0 = time.time()
    vectors = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"embedded {len(chunks)} chunks in {time.time() - t0:.1f}s")
    save_cache(chunks, vectors, fingerprint, model_name, chunk_size, overlap)
    return {"model_name": model_name, "chunks": chunks, "vectors": vectors,
            "fingerprint": fingerprint, "encoder": model}


def query_index(index: dict, encoder: SentenceTransformer, question: str,
                top_k: int = DEFAULT_TOP_K, use_prefix: bool | None = None) -> list[dict]:
    if not question.strip():
        raise ValueError("question must not be empty")
    if use_prefix is None:
        use_prefix = "bge" in index["model_name"].lower()

    q = (BGE_QUERY_PREFIX + question) if use_prefix else question
    q_vec = encoder.encode(q, normalize_embeddings=True)
    scores = index["vectors"] @ q_vec
    best = np.argsort(scores)[::-1][: min(top_k, len(scores))]
    return [{**index["chunks"][i], "score": float(scores[i])} for i in best]


def print_hits(hits: list[dict]) -> None:
    for rank, r in enumerate(hits, start=1):
        print(f"\n#{rank}  score {r['score']:.3f}  [{r['source']} | page {r['page']}]")
        print(r["text"][:300] + ("..." if len(r["text"]) > 300 else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic search over your corpus.")
    parser.add_argument("--folder", default="data", help="corpus folder (default: data)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"embedding model (default: {DEFAULT_MODEL})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rebuild", action="store_true", help="ignore the embedding cache and re-embed")
    parser.add_argument("--no-query-prefix", action="store_true",
                        help="disable the BGE instruction prefix (default: auto-detected from model name)")
    parser.add_argument("--question", help="ask a single question and exit (non-interactive)")
    args = parser.parse_args()

    index = create_index(args.folder, args.model,
                         args.chunk_size, args.overlap, args.rebuild)
    encoder = index["encoder"] or SentenceTransformer(args.model)
    use_prefix = False if args.no_query_prefix else None

    if args.question:
        print_hits(query_index(index, encoder, args.question, args.top_k, use_prefix))
        return 0

    print("Enter a question (or 'quit' to exit).")
    while True:
        try:
            q = input("\nAsk a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in ("quit", "exit", ""):
            break
        print_hits(query_index(index, encoder, q, args.top_k, use_prefix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
