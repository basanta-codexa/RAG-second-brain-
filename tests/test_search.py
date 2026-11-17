import numpy as np

from search import RERANK_MODEL, _build_hybrid_view, query_index


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
        self.scrolls = 0

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
        self.scrolls += 1
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


def _cached_index(chunks, vectors=None, extra=None):
    """An index as create_index would build it, with a prebuilt hybrid view."""
    vectors = _vectors() if vectors is None else vectors
    rows = [
        {"id": i, "origin": "test/x", **chunk, "vector": vec.tolist()}
        for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True))
    ]
    index = _index(chunks, extra)
    index["hybrid_cache"] = _build_hybrid_view(rows)
    return index


def test_pure_vector_search_uses_store():
    hits = query_index(_index(_chunks()), FakeEncoder(), "roof xyzzy", top_k=3)
    assert [h["source"] for h in hits] == ["a.md", "b.md", "c.md"]


def test_hybrid_uses_bm25_to_break_vector_ties():
    hits = query_index(_cached_index(_chunks()), FakeEncoder(), "roof xyzzy", top_k=1, hybrid=True)
    assert hits[0]["source"] == "a.md"
    assert "bm25_score" in hits[0]


def test_hybrid_does_not_expose_vectors_in_results():
    hits = query_index(_cached_index(_chunks()), FakeEncoder(), "roof", top_k=3, hybrid=True)
    assert all("vector" not in hit for hit in hits)


def test_hybrid_builds_cache_once_then_reuses_it():
    index = _index(_chunks())  # no prebuilt cache: exercises the fallback path
    assert index["store"].scrolls == 0
    query_index(index, FakeEncoder(), "roof", top_k=1, hybrid=True)
    assert index["store"].scrolls == 1
    query_index(index, FakeEncoder(), "roof", top_k=1, hybrid=True)
    assert index["store"].scrolls == 1  # second query must not re-fetch the corpus
    assert index["hybrid_cache"] is not None


def test_source_filtered_fallback_does_not_poison_the_global_cache():
    index = _index(_chunks())  # no prebuilt cache
    query_index(index, FakeEncoder(), "roof", top_k=3, hybrid=True, source="a.md")
    assert index["store"].scrolls == 1
    hits = query_index(index, FakeEncoder(), "roof", top_k=3, hybrid=True)
    assert {h["source"] for h in hits} == {"a.md", "b.md", "c.md"}
    assert index["store"].scrolls == 1


def test_cached_hybrid_source_filter_never_scrolls():
    index = _cached_index(_chunks())
    hits = query_index(index, FakeEncoder(), "roof xyzzy", top_k=3,
                       hybrid=True, source="a.md")
    assert [h["source"] for h in hits] == ["a.md"]
    assert "bm25_score" in hits[0]
    assert index["store"].scrolls == 0


def test_cross_encoder_reranker_reorders():
    class FlipReranker:
        def predict(self, pairs):
            return np.array([0.1, 0.1, 100.0])

    hits = query_index(_index(_chunks(), {"reranker": FlipReranker()}),
                       FakeEncoder(), "anything", top_k=1, rerank=True)
    assert hits[0]["source"] == "c.md"
    assert hits[0]["rerank_score"] == 100.0


def test_combined_hybrid_rerank_scores_the_fused_pool():
    class ZebraReranker:
        def __init__(self):
            self.pairs = None

        def predict(self, pairs):
            self.pairs = pairs
            return np.array([100.0 if "zebra" in text else 0.0 for _, text in pairs])

    reranker = ZebraReranker()
    index = _cached_index(_chunks(), extra={"reranker": reranker})
    hits = query_index(index, FakeEncoder(), "roof xyzzy", top_k=1,
                       hybrid=True, rerank=True)

    # The cross-encoder must actually run, over a pool wider than top_k=1.
    assert len(reranker.pairs) == 3
    assert len(hits) == 1
    assert hits[0]["source"] == "c.md"
    assert hits[0]["rerank_score"] == 100.0
    # Both stages are visible in the result, so the UI can show both scores.
    assert "bm25_score" in hits[0]
    assert "score" in hits[0]


def test_combined_mode_uses_a_wider_pool_than_hybrid_only():
    class CountingReranker:
        def __init__(self):
            self.sizes = []

        def predict(self, pairs):
            self.sizes.append(len(pairs))
            return np.zeros(len(pairs))

    reranker = CountingReranker()
    index = _cached_index(_chunks(), extra={"reranker": reranker})
    query_index(index, FakeEncoder(), "roof", top_k=1, hybrid=True)
    assert reranker.sizes == []  # hybrid-only must not load the cross-encoder

    query_index(index, FakeEncoder(), "roof", top_k=1, hybrid=True, rerank=True)
    assert reranker.sizes == [3]  # all candidates, not just top_k


def test_configured_reranker_model_is_used(monkeypatch):
    called = []

    class FakeReranker:
        def predict(self, pairs):
            return np.zeros(len(pairs))

    def fake_load(name):
        called.append(name)
        return FakeReranker()

    monkeypatch.setattr("search.load_reranker", fake_load)

    query_index(_index(_chunks()), FakeEncoder(), "roof", top_k=1, rerank=True)
    assert called == [RERANK_MODEL]

    called.clear()
    index = _index(_chunks(), extra={"reranker_model": "custom/reranker"})
    query_index(index, FakeEncoder(), "roof", top_k=1, rerank=True)
    assert called == ["custom/reranker"]


def test_source_filter_with_no_matches_returns_empty():
    hits = query_index(_index(_chunks()), FakeEncoder(), "roof", top_k=3, source="missing.md")
    assert hits == []
