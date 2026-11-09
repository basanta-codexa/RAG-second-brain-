import numpy as np

from search import query_index


class FakeEncoder:
    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


def _chunks():
    return [
        {"text": "mango cider roof apartment", "source": "a.md", "page": 1, "chunk_id": "a-1-c0"},
        {"text": "unrelated paper plain table", "source": "b.md", "page": 1, "chunk_id": "b-1-c0"},
        {"text": "zebra quantum console glasses", "source": "c.md", "page": 1, "chunk_id": "c-1-c0"},
    ]


def _vectors():
    return np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)


class FakeStore:
    """Mimics QdrantClient's scoring semantics exactly for the tests."""

    collection = "test"

    def __init__(self, chunks, vectors):
        self._chunks = chunks
        self._vectors = vectors

    def search(self, vector, limit, source=None):
        v = np.asarray(vector, dtype=np.float32)
        idx = [i for i, c in enumerate(self._chunks) if source is None or c["source"] == source]
        scored = [(float(self._vectors[i] @ v), i) for i in idx]
        scored.sort(key=lambda item: -item[0])
        return [
            {**self._chunks[i], "origin": "test/x", "score": score}
            for score, i in scored
        ][: min(limit, len(scored))]

    def count(self):
        return len(self._chunks)

    def scroll_all(self, with_vectors=False, limit=512, source=None):
        rows = []
        for i, chunk in enumerate(self._chunks):
            if source and chunk["source"] != source:
                continue
            row = {"id": i, "origin": "test/x", **chunk}
            if with_vectors:
                row["vector"] = self._vectors[i].tolist()
            rows.append(row)
        return rows


def _index(chunks, extra=None):
    index = {
        "model_name": "test-model",
        "chunks": chunks,
        "store": FakeStore(chunks, _vectors()),
    }
    if extra:
        index.update(extra)
    return index


def test_pure_vector_search_uses_store():
    hits = query_index(_index(_chunks()), FakeEncoder(), "roof xyzzy", top_k=3)
    assert [h["source"] for h in hits] == ["a.md", "b.md", "c.md"]


def test_hybrid_uses_bm25_to_break_vector_ties():
    from rank_bm25 import BM25Okapi

    from search import _tokenize

    chunks = _chunks()
    bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])
    hits = query_index(_index(chunks, {"bm25": bm25}),
                       FakeEncoder(), "roof xyzzy", top_k=1, hybrid=True)
    assert hits[0]["source"] == "a.md"
    assert "bm25_score" in hits[0]


def test_cross_encoder_reranker_reorders():
    class FlipReranker:
        def predict(self, pairs):
            return np.array([0.1, 0.1, 100.0])

    hits = query_index(_index(_chunks(), {"reranker": FlipReranker()}),
                       FakeEncoder(), "anything", top_k=1, rerank=True)
    assert hits[0]["source"] == "c.md"
    assert hits[0]["rerank_score"] == 100.0


def test_hybrid_source_filter_scopes_bm25_to_the_same_rows():
    hits = query_index(_index(_chunks()), FakeEncoder(), "roof xyzzy",
                       top_k=3, hybrid=True, source="a.md")
    assert [h["source"] for h in hits] == ["a.md"]
    assert "bm25_score" in hits[0]


def test_source_filter_with_no_matches_returns_empty():
    hits = query_index(_index(_chunks()), FakeEncoder(), "roof", top_k=3, source="missing.md")
    assert hits == []
