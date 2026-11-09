from evaluate import _single_page_sources, compute_metrics, find_rank


def test_find_rank_matches_origin_or_filename():
    results = [
        {"source": "a.md", "origin": "data/a.md", "page": 1},
        {"source": "b.md", "origin": "uploads/b.md", "page": 2},
    ]
    assert find_rank(results, [("b.md", 2)]) == 2
    assert find_rank(results, [("uploads/b.md", 2)]) == 2
    assert find_rank(results, [("b.md", 1)]) is None


def test_compute_metrics():
    m = compute_metrics([1, None, 2], top_k=3)
    assert m["n"] == 3
    assert m["hit1"] == 1
    assert m["hitk"] == 2
    assert m["mrr"] == (1.0 + 0.5) / 3


def test_single_page_sources():
    index = {"chunks": [
        {"source": "one.md", "origin": "data/one.md", "page": 1},
        {"source": "multi.md", "origin": "data/multi.md", "page": 1},
        {"source": "multi.md", "origin": "data/multi.md", "page": 2},
    ]}
    single = _single_page_sources(index)
    assert "one.md" in single and "data/one.md" in single
    assert "multi.md" not in single
