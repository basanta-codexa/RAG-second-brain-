import numpy as np

from search import query_index


class FakeEncoder:
    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            return np.array([1.0, 0.0])
        return np.array([[1.0, 0.0] for _ in texts])


def _chunks():
    return [
        {"text": "mango cider roof apartment", "source": "a.md", "page": 1, "chunk_id": "a-1-c0"},
        {"text": "unrelated paper plain table", "source": "b.md", "page": 1, "chunk_id": "b-1-c0"},
        {"text": "zebra quantum console glasses", "source": "c.md", "page": 1, "chunk_id": "c-1-c0"},
    ]


def _index(chunks, extra=None):
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])  # vector tie: order by index
    index = {"model_name": "test-model", "chunks": chunks, "vectors": vectors}
    if extra:
        index.update(extra)
    return index


def test_hybrid_uses_bm25_to_break_vector_ties():
    from rank_bm25 import BM25Okapi
    from search import _tokenize

    chunks = _chunks()
    bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])
    index = _index(chunks, {"bm25": bm25})

    hits = query_index(index, FakeEncoder(), "roof xyzzy", top_k=1, hybrid=True)
    assert hits[0]["source"] == "a.md"
    assert "bm25_score" in hits[0]


def test_hybrid_flag_off_keeps_pure_vector_order():
    index = _index(_chunks())  # no bm25 attached: must fall back cleanly
    hits = query_index(index, FakeEncoder(), "roof xyzzy", top_k=3, hybrid=False)
    assert [h["source"] for h in hits] == ["a.md", "b.md", "c.md"]


def test_cross_encoder_reranker_reorders():
    class FlipReranker:
        def predict(self, pairs):
            return np.array([0.1, 0.1, 100.0])

    chunks = _chunks()
    index = _index(chunks, {"reranker": FlipReranker()})
    hits = query_index(index, FakeEncoder(), "anything", top_k=1, rerank=True)
    assert hits[0]["source"] == "c.md"
    assert hits[0]["rerank_score"] == 100.0
