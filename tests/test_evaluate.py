import json

from evaluate import _single_page_sources, compute_metrics, evaluate, find_rank


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


def test_evaluate_writes_a_reproducible_report(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "one.md").write_text("x", encoding="utf-8")
    (data / "multi.md").write_text("x", encoding="utf-8")
    monkeypatch.setattr("evaluate.TESTS", [
        {"q": "question one?", "answers": [("one.md", 1)]},
        {"q": "question two?", "answers": [("multi.md", 2)]},
        {"q": "question three?", "answers": [("missing.md", 1)]},
    ])
    index = {
        "chunks": [
            {"source": "one.md", "origin": "data/one.md", "page": 1},
            {"source": "multi.md", "origin": "data/multi.md", "page": 1},
            {"source": "multi.md", "origin": "data/multi.md", "page": 2},
        ],
        "encoder": object(),
        "errors": [],
    }
    monkeypatch.setattr("evaluate.srch.create_index", lambda *a, **kw: index)

    def fake_query(index, encoder, question, top_k, **kwargs):
        source = "one.md" if "one" in question else "multi.md"
        return [{
            "source": source, "origin": f"data/{source}", "page": 1,
            "score": 1.0, "text": "evidence",
        }]

    monkeypatch.setattr("evaluate.srch.query_index", fake_query)
    out = tmp_path / "report.json"

    rc = evaluate(str(data), 3, "model", 800, 150, False, json_path=str(out))

    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["counts"] == {
        "total": 3, "evaluated": 2, "skipped": 1, "strict": 1, "lenient": 1,
    }
    assert report["overall"]["hit1"] == 1
    assert report["strict"]["hitk"] == 0
    assert report["skipped_questions"] == ["question three?"]
    assert report["config"]["model"] == "model"
    assert report["config"]["mode"] == "vector-only"


def test_compute_metrics_includes_confidence_interval():
    m = compute_metrics([1, None], top_k=3)
    low, high = m["hitk_ci95"]
    assert 0 <= low <= 0.5 <= high <= 1
    assert compute_metrics([], top_k=3)["hitk_ci95"] == [0.0, 0.0]


def test_single_page_sources():
    index = {"chunks": [
        {"source": "one.md", "origin": "data/one.md", "page": 1},
        {"source": "multi.md", "origin": "data/multi.md", "page": 1},
        {"source": "multi.md", "origin": "data/multi.md", "page": 2},
    ]}
    single = _single_page_sources(index)
    assert "one.md" in single and "data/one.md" in single
    assert "multi.md" not in single
