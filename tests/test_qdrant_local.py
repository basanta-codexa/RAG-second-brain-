"""End-to-end Store + create_index tests against qdrant-client's in-memory mode.

No Docker and no embedding model required: a keyword encoder provides
deterministic vectors, while the real Store, planner, Qdrant client, filters,
and state file are exercised.
"""
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient

import search
from qdrant_store import Store


class KeywordEncoder:
    """Tiny deterministic encoder: one axis per keyword, normalized."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        rows = []
        for text in texts:
            vec = np.zeros(8, dtype=np.float32)
            for axis, word in enumerate(("alpha", "beta", "gamma", "delta")):
                if word in text.lower():
                    vec[axis] = 1.0
            if not vec.any():
                vec[7] = 1.0
            if normalize_embeddings:
                norm = float(np.linalg.norm(vec))
                if norm:
                    vec /= norm
            rows.append(vec)
        return rows[0] if single else np.array(rows, dtype=np.float32)


def make_store(tmp_path: Path) -> Store:
    store = object.__new__(Store)  # bypass the live-server constructor
    store.url = ":memory:"
    store.collection = "integration"
    store.state_path = tmp_path / "state.json"
    store.client = QdrantClient(location=":memory:")
    return store


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(store: Store, data: Path, model: str = "m1", rebuild: bool = False):
    return search.create_index(str(data), model, chunk_size=50, overlap=10,
                               rebuild=rebuild, encoder=KeywordEncoder())


def test_in_memory_index_filters_updates_and_rebuilds(tmp_path: Path, monkeypatch):
    store = make_store(tmp_path)
    monkeypatch.setattr("search.Store", lambda *args, **kwargs: store)

    data = tmp_path / "data"
    write(data / "a.md", "alpha " * 30)
    write(data / "b.md", "beta " * 30)

    index = run(store, data)
    assert index["errors"] == []
    assert store.count() > 0
    assert store.state_path.exists()

    encoder = KeywordEncoder()
    only_a = search.query_index(index, encoder, "alpha", top_k=3, source="data/a.md")
    assert only_a and all(hit["source"] == "a.md" for hit in only_a)
    assert all(hit["origin"] == "data/a.md" for hit in only_a)

    only_b = search.query_index(index, encoder, "beta", top_k=3, source="b.md")
    assert only_b and all(hit["source"] == "b.md" for hit in only_b)

    assert search.query_index(index, encoder, "alpha", source="missing.md") == []

    hybrid = search.query_index(index, encoder, "alpha", top_k=3, hybrid=True)
    assert hybrid and "bm25_score" in hybrid[0]

    before = store.count()
    run(store, data)
    assert store.count() == before  # unchanged corpus does not duplicate points

    (data / "b.md").unlink()
    run(store, data)
    assert all(hit["source"] == "a.md" for hit in store.scroll_all())

    write(data / "a.md", "alpha. " * 3)  # fewer chunks: old extra ids must go
    run(store, data)
    after_shrink = store.count()
    run(store, data)
    assert store.count() == after_shrink

    index2 = run(store, data, model="m2")
    assert index2["errors"] == []
    assert store.count() == after_shrink  # rebuild did not double or 404
    assert "m2" in store.state_path.read_text(encoding="utf-8")


def test_in_memory_rebuild_flag_is_safe(tmp_path: Path, monkeypatch):
    store = make_store(tmp_path)
    monkeypatch.setattr("search.Store", lambda *args, **kwargs: store)

    data = tmp_path / "data"
    write(data / "a.md", "alpha " * 30)

    run(store, data)
    before = store.count()
    index = run(store, data, rebuild=True)
    assert index["errors"] == []
    assert store.count() == before
