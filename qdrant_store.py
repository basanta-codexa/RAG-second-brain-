"""Qdrant-backed vector store with per-file incremental indexing.

Requires a running Qdrant (docker):
  docker run -d --name second-brain-qdrant --restart unless-stopped \
      -p 6333:6333 -p 6334:6334 -v "<data-dir>:/qdrant/storage" qdrant/qdrant

State lives in .cache/qdrant_state.json — maps "<origin>/<file>" to the
sha256 and Qdrant point ids it produced. Any file whose hash changed is
re-embedded; deleted files have their points removed. Rebuild whole only
when the embedding model or the Qdrant collection disappears.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

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
STATE_PATH = Path(".cache") / "qdrant_state.json"
QDRANT_URL = os.getenv("SECOND_BRAIN_QDRANT", "http://127.0.0.1:6333")
_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "second-brain/chunks")


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


def plan(files: dict[str, str], state: dict | None,
         model_name: str, collection_exists: bool) -> dict:
    """Decide which files to embed, which points to drop, or a full rebuild.

    ``files`` maps file_key -> sha256. ``state`` is the persisted dict (or
    None). Pure function — unit tested without a Qdrant server.
    """
    if state is None:
        state = {}
    old = state.get("files", {})
    model_changed = state.get("model") != model_name
    collection_lost = bool(old) and not collection_exists

    if model_changed or collection_lost:
        drop_ids = [pid for f in old.values() for pid in f["point_ids"]]
        new_state = {"model": model_name, "files": {}, "collection_built": False}
        return {"to_embed": sorted(files), "drop_ids": drop_ids,
                "rebuild": True, "state": new_state}

    old_keys, new_keys = set(old), set(files)
    stale = old_keys - new_keys
    changed = {k for k in new_keys & old_keys if old[k]["sha256"] != files[k]}
    drop_keys = stale | changed
    drop_ids = [pid for k in drop_keys for pid in old[k]["point_ids"]]

    new_state = {"model": model_name, "files": dict(old), "collection_built": True}
    for k in drop_keys:
        new_state["files"].pop(k, None)

    to_embed = sorted((new_keys - old_keys) | changed)
    return {"to_embed": to_embed, "drop_ids": drop_ids,
            "rebuild": False, "state": new_state}


class Store:
    """Thin wrapper around QdrantClient with file-change planning built in."""

    def __init__(self, url: str | None = None, collection: str = COLLECTION,
                 state_path: Path | None = None) -> None:
        self.url = url or QDRANT_URL
        self.collection = collection
        self.state_path = Path(state_path) if state_path else STATE_PATH
        try:
            self.client = QdrantClient(url=self.url, timeout=10)
            self.client.get_collections()  # cheap connectivity check
        except Exception as exc:
            raise RuntimeError(
                f"Qdrant is not reachable at {self.url}. "
                "Start it: docker start second-brain-qdrant "
                f"(or recreate it — see README). Underlying error: {exc}"
            ) from exc

    # ---------- collection lifecycle ----------

    def exists(self) -> bool:
        try:
            return self.client.collection_exists(self.collection)
        except Exception:
            return False

    def create(self, vector_size: int) -> None:
        if self.exists():
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def drop(self) -> None:
        if self.exists():
            self.client.delete_collection(self.collection)

    # ---------- state ----------

    def load_state(self) -> dict | None:
        if not self.state_path.exists():
            return None
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def save_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    # ---------- writes ----------

    def delete_points(self, point_ids: list[str]) -> None:
        if not point_ids:
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=point_ids),
        )

    def delete_by_origin(self, origin: str) -> None:
        filt = Filter(must=[FieldCondition(key="origin", match=MatchValue(value=origin))])
        self.client.delete(collection_name=self.collection, points_selector=FilterSelector(filter=filt))

    def upsert(self, chunks: list[dict], vectors) -> list[str]:
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
            for pid, vec, c in zip(ids, vectors, chunks)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return ids

    # ---------- reads ----------

    def search(self, vector, limit: int):
        response = self.client.query_points(
            collection_name=self.collection,
            query=[float(x) for x in vector],
            limit=limit,
            with_payload=True,
        )
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
        return self.client.count(self.collection, exact=True).count

    def scroll_all(self, with_vectors: bool = False, limit: int = 512) -> list[dict]:
        points, next_offset = self.client.scroll(
            collection_name=self.collection,
            limit=limit,
            with_payload=True,
            with_vectors=with_vectors,
        )
        rows = [self._to_row(p, with_vectors) for p in points]
        while next_offset is not None:
            points, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=limit,
                offset=next_offset,
                with_payload=True,
                with_vectors=with_vectors,
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
