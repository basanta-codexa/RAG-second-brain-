"""Qdrant-backed vector store with per-file incremental indexing.

Requires a running Qdrant (docker):
  docker run -d --name second-brain-qdrant --restart unless-stopped \
      -p 6333:6333 -p 6334:6334 -v "<data-dir>:/qdrant/storage" qdrant/qdrant

State lives in .cache/qdrant_state.json — maps "<origin>/<file>" to the
sha256 and Qdrant point ids it produced. Any file whose hash changed is
re-embedded; deleted files have their points removed. Rebuild whole only
when the embedding model, the chunking configuration, or the state schema
changes, or when the Qdrant collection disappears.

The planner is a pure function so the dangerous parts can be tested without
a server. The store only performs I/O.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

COLLECTION = "second_brain"
SCHEMA_VERSION = 2
STATE_PATH = Path(".cache") / "qdrant_state.json"
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
QDRANT_URL = os.getenv("SECOND_BRAIN_QDRANT", DEFAULT_QDRANT_URL)
_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "second-brain/chunks")


class StoreUnavailable(RuntimeError):
    """Qdrant is not reachable; the caller should surface this as 503."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_key(folder: Path, path: Path) -> str:
    """Stable cross-machine identifier: "data/notes.md", "uploads/x.pdf"."""
    return f"{folder.name}/{path.name}"


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def index_signature(model_name: str, chunk_size: int, overlap: int) -> dict:
    """Everything that changes the meaning of a stored vector or chunk id."""
    return {
        "schema": SCHEMA_VERSION,
        "model": model_name,
        "chunk_size": chunk_size,
        "overlap": overlap,
    }


def _valid_state(state: dict | None) -> bool:
    if not isinstance(state, dict):
        return False
    files = state.get("files")
    if not isinstance(files, dict):
        return False
    for entry in files.values():
        if not isinstance(entry, dict):
            return False
        if not isinstance(entry.get("sha256"), str):
            return False
        ids = entry.get("point_ids")
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            return False
    return isinstance(state.get("signature"), dict)


def plan(files: dict[str, str], state: dict | None,
         signature: dict, collection_exists: bool) -> dict:
    """Decide which files to embed, which points to drop, or a full rebuild.

    ``files`` maps file_key -> sha256. ``state`` is the persisted dict (or
    None). ``signature`` is :func:`index_signature`. Pure function.

    ``drop_by_key`` maps the file key whose old points should be removed to
    those point ids. For changed files the caller must delete them only after
    the replacement points were upserted successfully.
    """
    if not _valid_state(state):
        state = None

    if state is None:
        return {
            "to_embed": sorted(files),
            "drop_by_key": {},
            "rebuild": True,
            "state": {"signature": signature, "files": {}},
        }

    old = state["files"]
    old_keys, new_keys = set(old), set(files)
    stale = old_keys - new_keys
    changed = {k for k in new_keys & old_keys if old[k]["sha256"] != files[k]}
    drop_keys = stale | changed

    drop_by_key = {k: list(old[k]["point_ids"]) for k in drop_keys}
    new_state = {"signature": signature, "files": dict(old)}
    for k in drop_keys:
        new_state["files"].pop(k, None)

    rebuild = state.get("signature") != signature or (bool(old) and not collection_exists)
    if rebuild:
        to_embed = sorted(files)
        new_state = {"signature": signature, "files": {}}
        # The collection, and every id in it, is about to disappear.
        drop_by_key = {}
    else:
        to_embed = sorted((new_keys - old_keys) | changed)

    return {
        "to_embed": to_embed,
        "drop_by_key": drop_by_key,
        "rebuild": rebuild,
        "state": new_state,
    }


class Store:
    """Thin wrapper around QdrantClient with file-change planning built in."""

    def __init__(self, url: str | None = None, collection: str = COLLECTION,
                 state_path: Path | None = None) -> None:
        self.url = url or os.getenv("SECOND_BRAIN_QDRANT", QDRANT_URL)
        self.collection = collection
        self.state_path = Path(state_path) if state_path else STATE_PATH
        try:
            self.client = QdrantClient(url=self.url, timeout=10)
            self.client.get_collections()  # cheap connectivity check
        except Exception as exc:
            raise StoreUnavailable(
                f"Qdrant is not reachable at {self.url}. "
                "Start it: docker start second-brain-qdrant "
                f"(or recreate it — see README). Underlying error: {exc}"
            ) from exc

    def _fail(self, action: str, exc: Exception):
        raise StoreUnavailable(
            f"Qdrant {action} failed at {self.url}: {exc}"
        ) from exc

    # ---------- collection lifecycle ----------

    def exists(self) -> bool:
        try:
            return self.client.collection_exists(self.collection)
        except Exception as exc:
            self._fail("connection check", exc)

    def create(self, vector_size: int) -> None:
        if self.exists():
            return
        try:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        except Exception as exc:
            self._fail("collection create", exc)

    def drop(self) -> None:
        if not self.exists():
            return
        try:
            self.client.delete_collection(self.collection)
        except Exception as exc:
            self._fail("collection drop", exc)

    # ---------- state ----------

    def load_state(self) -> dict | None:
        if not self.state_path.exists():
            return None
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return state if _valid_state(state) else None

    def save_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    # ---------- writes ----------

    def delete_points(self, point_ids: list[str]) -> None:
        if not point_ids or not self.exists():
            return
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=PointIdsList(points=point_ids),
            )
        except Exception as exc:
            self._fail("point delete", exc)

    def delete_by_origin(self, origin: str) -> None:
        if not self.exists():
            return
        filt = Filter(must=[FieldCondition(key="origin", match=MatchValue(value=origin))])
        try:
            self.client.delete(collection_name=self.collection,
                               points_selector=FilterSelector(filter=filt))
        except Exception as exc:
            self._fail("origin delete", exc)

    def upsert(self, chunks: list[dict], vectors) -> list[str]:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            if len(chunks) != 1:
                raise ValueError("one vector expected for one chunk")
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[0] != len(chunks):
            raise ValueError("chunks and vectors must have the same length")
        ids = [point_id(c["chunk_id"]) for c in chunks]
        points = [
            PointStruct(
                id=pid,
                vector=[float(x) for x in vec],
                payload={
                    "text": c["text"],
                    "source": c["source"],
                    "page": c["page"],
                    "chunk_id": c["chunk_id"],
                    "origin": c["origin"],
                },
            )
            for pid, vec, c in zip(ids, arr, chunks, strict=True)
        ]
        try:
            self.client.upsert(collection_name=self.collection, points=points)
        except Exception as exc:
            self._fail("upsert", exc)
        return ids

    # ---------- reads ----------

    @staticmethod
    def _source_filter(source: str | None) -> Filter | None:
        if not source:
            return None
        # Accept either an origin ("data/notes.md") or a bare filename
        # ("notes.md"). Origins are unique; filenames are not.
        return Filter(should=[
            FieldCondition(key="source", match=MatchValue(value=source)),
            FieldCondition(key="origin", match=MatchValue(value=source)),
        ])

    def search(self, vector, limit: int, source: str | None = None):
        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=np.asarray(vector, dtype=float).reshape(-1).tolist(),
                limit=limit,
                with_payload=True,
                query_filter=self._source_filter(source),
            )
        except Exception as exc:
            self._fail("search", exc)
        hits = []
        for point in response.points:
            p = point.payload or {}
            hits.append({
                "text": p.get("text", ""),
                "source": p.get("source", "?"),
                "page": int(p.get("page", -1)),
                "chunk_id": p.get("chunk_id", ""),
                "origin": p.get("origin", ""),
                "score": float(point.score),
            })
        return hits

    def count(self) -> int:
        if not self.exists():
            return 0
        try:
            return self.client.count(collection_name=self.collection, exact=True).count
        except Exception as exc:
            self._fail("count", exc)

    def _scroll(self, **kwargs):
        try:
            return self.client.scroll(collection_name=self.collection, **kwargs)
        except Exception as exc:
            self._fail("scroll", exc)

    def scroll_all(self, with_vectors: bool = False, limit: int = 512,
                   source: str | None = None) -> list[dict]:
        query_filter = self._source_filter(source)
        points, next_offset = self._scroll(
            limit=limit,
            with_payload=True,
            with_vectors=with_vectors,
            scroll_filter=query_filter,
        )
        rows = [self._to_row(p, with_vectors) for p in points]
        while next_offset is not None:
            points, next_offset = self._scroll(
                limit=limit,
                offset=next_offset,
                with_payload=True,
                with_vectors=with_vectors,
                scroll_filter=query_filter,
            )
            rows.extend(self._to_row(p, with_vectors) for p in points)
        return rows

    @staticmethod
    def _to_row(point, with_vectors: bool) -> dict:
        payload = point.payload or {}
        row = {
            "id": point.id,
            "text": payload.get("text", ""),
            "source": payload.get("source", "?"),
            "page": int(payload.get("page", -1)),
            "chunk_id": payload.get("chunk_id", ""),
            "origin": payload.get("origin", ""),
        }
        if with_vectors:
            row["vector"] = point.vector
        return row
