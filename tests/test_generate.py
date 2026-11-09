import pytest

from app.generate import (
    FALLBACK_REFUSAL,
    GenerationError,
    answer,
    invalid_citations,
    is_refusal,
    pack_passages,
)


def _hit(i: int, text: str = "passage text") -> dict:
    return {"text": text, "source": "notes.md", "page": i,
            "chunk_id": f"c{i}", "score": 0.9}


def test_pack_passages_keeps_ranks_and_a_prefix():
    hits = [_hit(i, "x" * 2000) for i in range(1, 11)]
    passages, used = pack_passages(hits)
    assert 0 < len(used) < len(hits)
    assert [p["id"] for p in passages] == list(range(1, len(passages) + 1))
    assert [h["rank"] for h in used] == list(range(1, len(used) + 1))
    assert used[0]["text"].startswith("x")


def test_pack_passages_truncates_long_text():
    hits = [_hit(1, "y" * 5000)]
    passages, used = pack_passages(hits)
    assert len(passages[0]["text"]) == 1200
    assert used[0]["text"] == passages[0]["text"]
    assert len(used) == 1


def test_refusal_detection_tolerates_case_and_punctuation():
    assert is_refusal(FALLBACK_REFUSAL)
    assert is_refusal("I don't know - the documents don't answer this.")
    assert is_refusal("  i don’t know — the documents don’t answer this.  ")
    assert is_refusal("I don't know — the documents don't answer this question.")
    assert not is_refusal("The documents say it is Oslo. [1]")


def test_invalid_citations():
    assert invalid_citations("ok [1] and [2]", 2) == []
    assert invalid_citations("ok [1], [0], [9]", 3) == [0, 9]


def test_answer_without_hits_refuses_locally():
    result = answer("anything", [])
    assert result["refused"] is True
    assert result["answer"] == FALLBACK_REFUSAL
    assert result["used"] == []


def test_missing_key_is_a_503(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(GenerationError) as exc:
        answer("anything", [_hit(1)])
    assert exc.value.status_code == 503
