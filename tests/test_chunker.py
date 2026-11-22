import re

import pytest

from chunker import chunk_records, chunk_text, chunk_text_tokens


class WordTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]}


def test_empty_input_produces_no_chunks():
    assert chunk_text("") == []


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=150)


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=0, overlap=0)


def test_no_text_is_dropped():
    text = "Sentence one. Sentence two. Sentence three. " * 40
    chunks = chunk_text(text, chunk_size=300, overlap=150)
    assert chunks
    for chunk in chunks:
        assert chunk.strip() == chunk and chunk


def test_cuts_at_question_and_exclamation_marks():
    text = "a" * 40 + "? " + "b" * 40 + "! " + "c" * 40 + ". " + "d" * 40
    chunks = chunk_text(text, chunk_size=60, overlap=10)
    assert chunks[0].endswith("?")
    assert any(chunk.endswith("!") for chunk in chunks)


def test_large_overlap_still_terminates():
    chunks = chunk_text("abcdefghij" * 20, chunk_size=10, overlap=9)
    assert 0 < len(chunks) < 1000


def test_token_chunks_are_whole_tokens_and_overlap():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text_tokens(text, WordTokenizer(), token_size=20, token_overlap=5)
    assert len(chunks) > 1
    words = set(text.split())
    for chunk in chunks:
        assert chunk == chunk.strip() and chunk
        assert set(chunk.split()) <= words
    assert chunks[0] != chunks[-1]


def test_token_parameters_are_validated():
    with pytest.raises(ValueError):
        chunk_text_tokens("x y", WordTokenizer(), token_size=0)
    with pytest.raises(ValueError):
        chunk_text_tokens("x y", WordTokenizer(), token_size=10, token_overlap=10)


def test_token_mode_requires_a_tokenizer():
    with pytest.raises(ValueError, match="tokenizer"):
        chunk_records([{"text": "x", "source": "a.md", "page": 1}], chunk_mode="token")


def test_token_mode_changes_the_index_signature():
    from qdrant_store import index_signature

    char = index_signature("m", 800, 150)
    token = index_signature("m", 800, 150, "token", 256, 48)
    assert "chunk_mode" not in char
    assert token["chunk_mode"] == "token"
    assert token["token_size"] == 256
