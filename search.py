"""Semantic search over the corpus, backed by Qdrant (Docker).

Requires a running Qdrant — see README. Indexing is incremental: each file is
content-hashed, only new/changed files are embedded, deleted files have their
vectors removed. State is tracked in .cache/qdrant_state.json.

Usage:
    python search.py                    # interactive REPL
    python search.py --question "..."   # one-shot answer
    python search.py --rebuild          # drop and re-embed everything
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from rank_bm25 import BM25Okapi

from chunker import chunk_records
from loader import SUPPORTED_SUFFIXES, load_pdf, load_text
from qdrant_store import (
    Store,
    StoreUnavailable,
    file_key,
    index_signature,
    plan,
    sha256_file,
)

if TYPE_CHECKING:  # sentence-transformers pulls in torch; keep imports lazy
    from sentence_transformers import CrossEncoder, SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_CHUNK_SIZE = 800      # BGE-en reads ~512 tokens; keep chunks modest
DEFAULT_OVERLAP = 150
DEFAULT_TOP_K = 3
PROGRESS_PATH = Path(".cache") / "index_progress.json"

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_encoder(model_name: str) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def load_reranker() -> CrossEncoder:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _write_progress(**fields) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = {"at": round(time.time(), 3), **fields}
    tmp = PROGRESS_PATH.with_suffix(PROGRESS_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(fields), encoding="utf-8")
    tmp.replace(PROGRESS_PATH)


def _clear_progress() -> None:
    try:
        PROGRESS_PATH.unlink()
    except FileNotFoundError:
        pass


def _target_files(folders: list[str]) -> dict[str, Path]:
    targets: dict[str, Path] = {}
    for folder_name in folders:
        root = Path(folder_name)
        if not root.is_dir():
            raise FileNotFoundError(f"corpus folder not found: {root}")
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                key = file_key(root, path)
                if key in targets and targets[key] != path:
                    raise ValueError(
                        f"two corpus folders share the key {key!r}; "
                        "give the folders distinct names"
                    )
                targets[key] = path
    return targets


def _load_one(path: Path, origin: str) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        records = load_pdf(path)
    else:
        records = load_text(path)
    for record in records:
        record["origin"] = origin
        record["source"] = path.name
    return records


def _embed_files(store: Store, files: list[tuple[str, Path]], model_name: str,
                 chunk_size: int, overlap: int,
                 encoder: SentenceTransformer | None = None
                 ) -> tuple[dict[str, dict], list[dict], Any]:
    """Read, chunk, embed and upsert. A broken file is reported, not fatal."""
    if not files:
        return {}, [], encoder

    # Phase 1: read + chunk everything first, so the total is known up front
    prepared = []
    errors: list[dict] = []
    total_chunks = 0
    for i, (key, path) in enumerate(files, start=1):
        _write_progress(active=True, stage="reading", file_index=i,
                        files_total=len(files), current_file=key,
                        chunks_done=0, chunks_total=None)
        try:
            records = _load_one(path, key)
            hashed = sha256_file(path)
            chunks = chunk_records(records, chunk_size, overlap)
        except Exception as exc:  # one unreadable file must not kill the corpus
            errors.append({"file": key, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  {key}: skipped ({type(exc).__name__}: {exc})", file=sys.stderr)
            continue
        total_chunks += len(chunks)
        prepared.append((key, hashed, chunks))

    if not prepared:
        return {}, errors, encoder

    print(f"embedding {total_chunks} chunks from {len(prepared)} file(s) with {model_name} ...")
    model = encoder or load_encoder(model_name)
    done: dict[str, dict] = {}
    embedded_so_far = 0
    for i, (key, hashed, chunks) in enumerate(prepared, start=1):
        if not chunks:
            print(f"  {key}: nothing readable, skipped")
            done[key] = {"sha256": hashed, "point_ids": []}
            continue
        _write_progress(active=True, stage="embedding", file_index=i,
                        files_total=len(prepared), current_file=key,
                        chunks_done=embedded_so_far, chunks_total=total_chunks)
        t0 = time.time()
        try:
            vectors = np.asarray(
                model.encode(
                    [c["text"] for c in chunks],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
            if vectors.ndim == 1:  # some encoders collapse a single-item batch
                vectors = vectors.reshape(1, -1)
            store.create(vector_size=vectors.shape[1])
            ids = store.upsert(chunks, vectors)
        except StoreUnavailable:
            raise  # backend outage: abort the sync, keep the state untouched
        except Exception as exc:
            errors.append({"file": key, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  {key}: FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)
            continue
        embedded_so_far += len(chunks)
        done[key] = {"sha256": hashed, "point_ids": ids}
        print(f"  {key}: {len(chunks)} chunks in {time.time() - t0:.1f}s")
    return done, errors, model


def create_index(folder: str | list[str] = "data", model_name: str = DEFAULT_MODEL,
                 chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP,
                 rebuild: bool = False,
                 encoder: SentenceTransformer | None = None) -> dict:
    """Sync the Qdrant collection with the current corpus files.

    ``folder`` may be one folder or a list; all supported files are merged.
    Only new/changed files are (re-)embedded. Unchanged files are left alone.

    Returns the index plus ``errors``: files that could not be read or embedded.
    Those files keep their previous points and will be retried next run.
    """
    folders = [folder] if isinstance(folder, str) else [str(f) for f in folder]
    if not folders:
        raise ValueError("at least one corpus folder is required")

    store = Store()
    files = _target_files(folders)
    if not files:
        raise RuntimeError(f"no documents in {', '.join(folders)}; is everything empty?")

    state = None if rebuild else store.load_state()
    signature = index_signature(model_name, chunk_size, overlap)
    plan_result = plan({k: sha256_file(p) for k, p in files.items()},
                       state, signature, store.exists())

    if plan_result["rebuild"]:
        store.drop()

    # Files that no longer exist can be removed immediately: there is no
    # replacement content at risk. Changed files are handled after upsert.
    to_embed_keys = plan_result["to_embed"]
    for key, old_ids in plan_result["drop_by_key"].items():
        if key not in to_embed_keys:
            store.delete_points(old_ids)

    _clear_progress()  # drop stale progress from any crashed run
    try:
        updates, errors, encoder_used = _embed_files(
            store, [(k, files[k]) for k in to_embed_keys],
            model_name, chunk_size, overlap, encoder,
        )
    finally:
        _clear_progress()

    # Only now is it safe to remove the old points of changed files: their
    # replacements are already in Qdrant. A failed file keeps its old points.
    for key, entry in updates.items():
        old_ids = plan_result["drop_by_key"].get(key, [])
        new_ids = set(entry["point_ids"])
        store.delete_points([pid for pid in old_ids if pid not in new_ids])

    new_state = plan_result["state"]
    new_state["files"].update(updates)
    # A changed file that failed keeps its old points. Put its old state entry
    # back too, so the next run still sees a hash mismatch and retries it.
    if state and not plan_result["rebuild"]:
        for key in to_embed_keys:
            if key not in updates and key in state["files"]:
                new_state["files"][key] = state["files"][key]
    store.save_state(new_state)

    chunks = store.scroll_all(with_vectors=False) if store.exists() else []
    reused = max(0, len(chunks) - sum(len(v["point_ids"]) for v in updates.values()))
    print(f"collection {store.collection}: {store.count()} points "
          f"({len(updates)} file(s) re-embedded, {reused} chunk(s) reused)")
    if errors:
        print(f"warning: {len(errors)} file(s) could not be indexed: "
              f"{', '.join(e['file'] for e in errors)}", file=sys.stderr)

    return {
        "model_name": model_name,
        "chunks": chunks,
        "encoder": encoder_used,
        "store": store,
        "errors": errors,
        "embedded": bool(updates),
    }


# ---------- query ----------

_RRF_K = 60
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_POOL = 20


def _rrf_order(vector_scores: np.ndarray, bm25_scores: np.ndarray) -> np.ndarray:
    def ranks(scores):
        order = np.argsort(-scores, kind="stable")
        r = np.empty(len(order), dtype=np.int64)
        r[order] = np.arange(len(order))
        return r

    fused = 1.0 / (_RRF_K + ranks(vector_scores) + 1) \
        + 1.0 / (_RRF_K + ranks(bm25_scores) + 1)
    return np.argsort(-fused, kind="stable")


def query_index(index: dict, encoder: SentenceTransformer, question: str,
                top_k: int = DEFAULT_TOP_K, use_prefix: bool | None = None,
                hybrid: bool = False, rerank: bool = False,
                source: str | None = None) -> list[dict]:
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    model_name = index["model_name"]
    if use_prefix is None:
        use_prefix = "bge" in model_name.lower()
    q = (BGE_QUERY_PREFIX + question) if use_prefix else question
    q_vec = np.asarray(encoder.encode(q, normalize_embeddings=True),
                       dtype=np.float32).reshape(-1)

    store: Store = index["store"]
    total = len(index["chunks"])
    if total == 0:
        raise RuntimeError("collection empty — nothing to search")
    k = min(top_k, total)

    if hybrid:
        rows = store.scroll_all(with_vectors=True, source=source)
        if not rows:
            return []
        bm25 = BM25Okapi([_tokenize(r["text"]) for r in rows])
        vectors = np.array([r["vector"] for r in rows], dtype=np.float32)
        vector_scores = vectors @ q_vec
        bm25_scores = np.array(bm25.get_scores(_tokenize(question)))
        order = _rrf_order(vector_scores, bm25_scores)[:k]
        return [
            {**{key: val for key, val in rows[i].items() if key != "vector"},
             "score": float(vector_scores[i]),
             "bm25_score": float(bm25_scores[i])}
            for i in order
        ]

    if rerank:
        pool = store.search(q_vec, limit=min(RERANK_POOL, total), source=source)
        if not pool:
            return []
        reranker = index.get("reranker")
        if reranker is None:
            reranker = load_reranker()
            index["reranker"] = reranker
        try:
            r_scores = np.asarray(
                reranker.predict([(question, h["text"]) for h in pool]), dtype=float)
        except Exception as exc:
            raise RuntimeError(f"reranker unavailable ({RERANK_MODEL}): {exc}") from exc
        order = np.argsort(-r_scores, kind="stable")[:k]
        return [{**pool[j], "rerank_score": float(r_scores[j])} for j in order]

    return store.search(q_vec, limit=k, source=source)


def print_hits(hits: list[dict]) -> None:
    for rank, r in enumerate(hits, start=1):
        parts = [f"vector {r['score']:.3f}"]
        if "bm25_score" in r:
            parts.append(f"bm25 {r['bm25_score']:.2f}")
        if "rerank_score" in r:
            parts.append(f"rerank {r['rerank_score']:.3f}")
        print(f"\n#{rank}  {'  '.join(parts)}  [{r['source']} | page {r['page']}]")
        print(r["text"][:300] + ("..." if len(r["text"]) > 300 else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic search over your corpus (Qdrant).")
    parser.add_argument("--folder", default="data")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"embedding model (default: {DEFAULT_MODEL})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rebuild", action="store_true", help="drop the collection and re-embed everything")
    parser.add_argument("--no-query-prefix", action="store_true",
                        help="disable the BGE instruction prefix")
    parser.add_argument("--hybrid", action="store_true", help="BM25+vector RRF fusion")
    parser.add_argument("--rerank", action="store_true",
                        help=f"re-rank with cross-encoder ({RERANK_MODEL})")
    parser.add_argument("--question", help="ask one question and exit")
    parser.add_argument("--source", help="restrict retrieval to one origin or filename (e.g. data/notes.md)")
    args = parser.parse_args()

    try:
        index = create_index(args.folder, args.model,
                             args.chunk_size, args.overlap, args.rebuild)
        encoder = index["encoder"] or load_encoder(args.model)
        use_prefix = False if args.no_query_prefix else None

        if args.question:
            print_hits(query_index(index, encoder, args.question, args.top_k, use_prefix,
                                   hybrid=args.hybrid, rerank=args.rerank, source=args.source))
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
            print_hits(query_index(index, encoder, q, args.top_k, use_prefix,
                                   hybrid=args.hybrid, rerank=args.rerank, source=args.source))
        return 0
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
