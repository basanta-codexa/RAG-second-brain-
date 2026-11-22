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
import os
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from dotenv import load_dotenv
from filelock import Timeout
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

load_dotenv(Path(__file__).resolve().with_name(".env"))

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_CHUNK_SIZE = 800      # BGE-en reads ~512 tokens; keep chunks modest
DEFAULT_OVERLAP = 150
DEFAULT_TOP_K = 3
DEFAULT_CHUNK_MODE = "char"
DEFAULT_TOKEN_SIZE = 256       # BGE-en is trained for ~512 tokens; 256 is safe
DEFAULT_TOKEN_OVERLAP = 48
PROGRESS_PATH = Path(".cache") / "index_progress.json"

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_encoder(model_name: str) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    # A cached model must not trigger HF Hub checks: on this machine that costs
    # ~200s per load. Fall back to a normal load only when nothing is cached.
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


def load_reranker(model_name: str | None = None) -> CrossEncoder:
    from sentence_transformers import CrossEncoder

    name = model_name or RERANK_MODEL
    try:
        return CrossEncoder(name, local_files_only=True)
    except Exception:
        return CrossEncoder(name)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _matches_source(row: dict, source: str | None) -> bool:
    """Mirror Store._source_filter: accept an origin or a bare filename."""
    return not source or row.get("source") == source or row.get("origin") == source


def _build_hybrid_view(rows: list[dict]) -> dict:
    """Precompute the BM25 index and vector matrix for a fixed set of rows.

    Built once per index sync (or once per query for indexes that predate the
    cache) so a query no longer drags the whole corpus over HTTP. The cache is
    owned by the index dict, which the web app already keeps per process.
    """
    matrix = np.asarray([r["vector"] for r in rows], dtype=np.float32)
    public = [{k: v for k, v in r.items() if k != "vector"} for r in rows]
    bm25 = BM25Okapi([_tokenize(r["text"]) for r in public])
    return {"rows": public, "matrix": matrix, "bm25": bm25, "by_source": {}}


def _subset_view(view: dict, indices: list[int]) -> dict | None:
    """Rebuild a view over a subset, so source filters keep exact BM25 IDF."""
    if not indices:
        return None
    rows = [view["rows"][i] for i in indices]
    return {
        "rows": rows,
        "matrix": view["matrix"][indices],
        "bm25": BM25Okapi([_tokenize(r["text"]) for r in rows]),
        "by_source": {},
    }


def _hybrid_view(index: dict, source: str | None = None) -> dict | None:
    """Return a cached hybrid view, scoped to *source* when requested."""
    view = index.get("hybrid_cache")
    if view is None:
        # Always build the global view first. Filtering inside the scroll would
        # cache a single source as if it were the whole corpus and silently
        # poison every later unfiltered query.
        rows = index["store"].scroll_all(with_vectors=True)
        if not rows or any("vector" not in r for r in rows):
            return None
        view = _build_hybrid_view(rows)
        index["hybrid_cache"] = view  # self-cache for callers that skip create_index
    if not source:
        return view
    by_source = view.setdefault("by_source", {})
    if source not in by_source:
        indices = [i for i, r in enumerate(view["rows"]) if _matches_source(r, source)]
        by_source[source] = _subset_view(view, indices)
    return by_source[source]


def _hybrid_candidates(index: dict, q_vec: np.ndarray, question: str,
                       limit: int, source: str | None = None) -> list[dict]:
    """RRF-fuse vector and BM25 scores over the cached corpus view."""
    view = _hybrid_view(index, source)
    if view is None:
        return []
    rows = view["rows"]
    vector_scores = view["matrix"] @ q_vec
    bm25_scores = np.asarray(view["bm25"].get_scores(_tokenize(question)), dtype=float)
    order = _rrf_order(vector_scores, bm25_scores)[: min(limit, len(rows))]
    return [
        {**rows[i], "score": float(vector_scores[i]), "bm25_score": float(bm25_scores[i])}
        for i in order
    ]


def _rerank_candidates(index: dict, question: str, pool: list[dict], top_k: int) -> list[dict]:
    """Cross-encoder reorder of an already-retrieved candidate pool."""
    if not pool:
        return []
    reranker_model = index.get("reranker_model") or RERANK_MODEL
    reranker = index.get("reranker")
    if reranker is None:
        reranker = load_reranker(reranker_model)
        index["reranker"] = reranker
    try:
        scores = np.asarray(
            reranker.predict([(question, h["text"]) for h in pool]), dtype=float)
    except Exception as exc:
        raise RuntimeError(f"reranker unavailable ({reranker_model}): {exc}") from exc
    order = np.argsort(-scores, kind="stable")[: min(top_k, len(pool))]
    return [{**pool[j], "rerank_score": float(scores[j])} for j in order]


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
                 encoder: SentenceTransformer | None = None,
                 chunk_mode: str = DEFAULT_CHUNK_MODE,
                 token_size: int = DEFAULT_TOKEN_SIZE,
                 token_overlap: int = DEFAULT_TOKEN_OVERLAP,
                 ) -> tuple[dict[str, dict], list[dict], Any]:
    """Read, chunk, embed and upsert. A broken file is reported, not fatal."""
    if not files:
        return {}, [], encoder

    # Load once here: token chunking needs the model's tokenizer up front.
    model = encoder or load_encoder(model_name)
    tokenizer = getattr(model, "tokenizer", None) if chunk_mode == "token" else None

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
            chunks = chunk_records(
                records, chunk_size, overlap,
                chunk_mode=chunk_mode, tokenizer=tokenizer,
                token_size=token_size, token_overlap=token_overlap,
            )
        except Exception as exc:  # one unreadable file must not kill the corpus
            errors.append({"file": key, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  {key}: skipped ({type(exc).__name__}: {exc})", file=sys.stderr)
            continue
        total_chunks += len(chunks)
        prepared.append((key, hashed, chunks))

    if not prepared:
        return {}, errors, encoder

    print(f"embedding {total_chunks} chunks from {len(prepared)} file(s) with {model_name} ...")
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


class IndexBusy(RuntimeError):
    """Another process holds the index lock; the caller should retry."""


def create_index(folder: str | list[str] = "data", model_name: str = DEFAULT_MODEL,
                 chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP,
                 rebuild: bool = False,
                 encoder: SentenceTransformer | None = None,
                 reranker_model: str | None = None,
                 lock_timeout: float | None = None,
                 chunk_mode: str = DEFAULT_CHUNK_MODE,
                 token_size: int = DEFAULT_TOKEN_SIZE,
                 token_overlap: int = DEFAULT_TOKEN_OVERLAP) -> dict:
    """Sync the Qdrant collection with the current corpus files.

    ``folder`` may be one folder or a list; all supported files are merged.
    Only new/changed files are (re-)embedded. Unchanged files are left alone.

    A cross-process lock serializes the web app, CLI, and evaluation script, so
    two syncs cannot plan against the same state and overwrite each other.

    Returns the index plus ``errors``: files that could not be read or embedded.
    Those files keep their previous points and will be retried next run.
    """
    folders = [folder] if isinstance(folder, str) else [str(f) for f in folder]
    if not folders:
        raise ValueError("at least one corpus folder is required")

    store = Store()
    try:
        with store.lock(timeout=lock_timeout):
            return _sync_index(store, folders, model_name, chunk_size, overlap,
                               rebuild, encoder, reranker_model,
                               chunk_mode, token_size, token_overlap)
    except Timeout as exc:
        raise IndexBusy(
            f"another index sync is running (lock: {store.lock_path()}); "
            "wait for it to finish, or remove the lock file if it is stale"
        ) from exc


def _sync_index(store: Store, folders: list[str], model_name: str,
                chunk_size: int, overlap: int, rebuild: bool,
                encoder: SentenceTransformer | None,
                reranker_model: str | None,
                chunk_mode: str = DEFAULT_CHUNK_MODE,
                token_size: int = DEFAULT_TOKEN_SIZE,
                token_overlap: int = DEFAULT_TOKEN_OVERLAP) -> dict:
    files = _target_files(folders)
    if not files:
        raise RuntimeError(f"no documents in {', '.join(folders)}; is everything empty?")

    state = None if rebuild else store.load_state()
    signature = index_signature(model_name, chunk_size, overlap,
                                chunk_mode, token_size, token_overlap)
    plan_result = plan({k: sha256_file(p) for k, p in files.items()},
                       state, signature, store.exists())

    if plan_result["rebuild"]:
        # Invalidate state before dropping: an interrupted rebuild must not
        # leave stale point ids that convince the next sync to skip repairs.
        store.clear_state()
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
            chunk_mode, token_size, token_overlap,
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

    # One scroll per sync builds both the public chunk list and the reusable
    # hybrid view. Queries then score in-process instead of re-fetching every
    # vector and payload over HTTP.
    rows = store.scroll_all(with_vectors=True) if store.exists() else []
    chunks = [{k: v for k, v in row.items() if k != "vector"} for row in rows]
    hybrid_cache = (
        _build_hybrid_view(rows) if rows and all("vector" in r for r in rows) else None
    )
    reused = max(0, len(chunks) - sum(len(v["point_ids"]) for v in updates.values()))
    print(f"collection {store.collection}: {store.count()} points "
          f"({len(updates)} file(s) re-embedded, {reused} chunk(s) reused)")
    if errors:
        print(f"warning: {len(errors)} file(s) could not be indexed: "
              f"{', '.join(e['file'] for e in errors)}", file=sys.stderr)

    return {
        "model_name": model_name,
        "reranker_model": reranker_model or RERANK_MODEL,
        "chunks": chunks,
        "hybrid_cache": hybrid_cache,
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
        # A combined query needs a wider fusion pool for the cross-encoder to
        # reorder; hybrid-only can stop at k directly.
        pool = _hybrid_candidates(index, q_vec, question,
                                  RERANK_POOL if rerank else k, source)
        if rerank:
            return _rerank_candidates(index, question, pool, k)
        return pool

    if rerank:
        pool = store.search(q_vec, limit=min(RERANK_POOL, total), source=source)
        return _rerank_candidates(index, question, pool, k)

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
    parser.add_argument("--folder", default=os.getenv("SECOND_BRAIN_DATA", "data"))
    parser.add_argument("--model", default=os.getenv("SECOND_BRAIN_MODEL", DEFAULT_MODEL),
                        help=f"embedding model (default: {DEFAULT_MODEL})")
    parser.add_argument("--chunk-size", type=int,
                        default=int(os.getenv("SECOND_BRAIN_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)))
    parser.add_argument("--overlap", type=int,
                        default=int(os.getenv("SECOND_BRAIN_OVERLAP", DEFAULT_OVERLAP)))
    parser.add_argument("--chunk-mode", choices=("char", "token"),
                        default=os.getenv("SECOND_BRAIN_CHUNK_MODE", DEFAULT_CHUNK_MODE),
                        help="character-based (default) or tokenizer-based chunks")
    parser.add_argument("--token-size", type=int,
                        default=int(os.getenv("SECOND_BRAIN_TOKEN_SIZE", DEFAULT_TOKEN_SIZE)))
    parser.add_argument("--token-overlap", type=int,
                        default=int(os.getenv("SECOND_BRAIN_TOKEN_OVERLAP",
                                              DEFAULT_TOKEN_OVERLAP)))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rebuild", action="store_true", help="drop the collection and re-embed everything")
    parser.add_argument("--no-query-prefix", action="store_true",
                        help="disable the BGE instruction prefix")
    parser.add_argument("--hybrid", action="store_true", help="BM25+vector RRF fusion")
    parser.add_argument("--rerank", action="store_true",
                        help="re-rank with a cross-encoder")
    parser.add_argument("--reranker", default=os.getenv("SECOND_BRAIN_RERANKER"),
                        help=f"cross-encoder model id for reranking (default: {RERANK_MODEL})")
    parser.add_argument("--question", help="ask one question and exit")
    parser.add_argument("--source", help="restrict retrieval to one origin or filename (e.g. data/notes.md)")
    args = parser.parse_args()

    try:
        index = create_index(args.folder, args.model,
                             args.chunk_size, args.overlap, args.rebuild,
                             reranker_model=args.reranker,
                             chunk_mode=args.chunk_mode, token_size=args.token_size,
                             token_overlap=args.token_overlap)
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
