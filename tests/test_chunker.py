import pytest

from chunker import chunk_text


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
