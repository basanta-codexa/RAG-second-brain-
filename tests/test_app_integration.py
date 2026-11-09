"""End-to-end API test with real FastAPI routes and in-memory Qdrant.

Only the embedding model is faked. Indexing, filtering, the index cache,
uploads, and deletion are all real.
"""
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.main import create_app
from qdrant_store import Store


class KeywordEncoder:
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
    store = object.__new__(Store)
    store.url = ":memory:"
    store.collection = "app-integration"
    store.state_path = tmp_path / "state.json"
    store.client = QdrantClient(location=":memory:")
    return store


def test_full_request_path(tmp_path: Path, monkeypatch):
    store = make_store(tmp_path)
    encoder = KeywordEncoder()
    monkeypatch.setattr("search.Store", lambda *args, **kwargs: store)
    monkeypatch.setattr("search.load_encoder", lambda model: encoder)

    data = tmp_path / "data"
    app = create_app(
        data_dir=str(data),
        upload_dir=str(tmp_path / "uploads"),
        model_name="test-model",
    )
    (data / "a.md").write_text("alpha is the first letter. " * 10, encoding="utf-8")
    client = TestClient(app)

    status = client.get("/api/status").json()
    assert status["sources"] == ["data/a.md"]

    first = client.post("/api/search", json={
        "question": "alpha", "top_k": 3, "source": "data/a.md",
    })
    assert first.status_code == 200
    body = first.json()
    assert body["hits"] and body["hits"][0]["source"] == "a.md"
    assert body["hits"][0]["origin"] == "data/a.md"
    assert body["cached_index"] is False

    second = client.post("/api/search", json={"question": "alpha", "top_k": 3}).json()
    assert second["cached_index"] is True

    upload = client.post("/api/upload", files={
        "file": ("notes.txt", b"beta is the second letter.", "text/plain"),
    })
    assert upload.status_code == 200
    assert upload.json()["name"] == "notes.txt"

    filtered = client.post("/api/search", json={
        "question": "beta", "top_k": 3, "source": "uploads/notes.txt",
    }).json()
    assert filtered["hits"] and filtered["hits"][0]["source"] == "notes.txt"
    assert filtered["hits"][0]["origin"] == "uploads/notes.txt"
    assert "uploads/notes.txt" in client.get("/api/status").json()["sources"]

    assert client.delete("/api/uploads/notes.txt").status_code == 200
    after = client.post("/api/search", json={"question": "beta", "top_k": 3}).json()
    assert all(hit["source"] != "notes.txt" for hit in after["hits"])
    assert "uploads/notes.txt" not in client.get("/api/status").json()["sources"]
