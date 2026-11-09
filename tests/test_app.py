from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import create_app, sanitize_filename
from qdrant_store import StoreUnavailable


@pytest.fixture()
def client(tmp_path: Path):
    data_dir = tmp_path / "data"
    upload_dir = tmp_path / "uploads"
    app = create_app(
        data_dir=str(data_dir),
        upload_dir=str(upload_dir),
        model_name="test-model",
        max_mb=1,
    )
    return TestClient(app)


def _fake_hits(text: str = "copper vault Oslo archives answer text") -> list[dict]:
    return [{
        "text": text, "source": "notes.txt", "page": 1,
        "chunk_id": "notes.txt-p1-c0", "origin": "uploads/notes.txt", "score": 0.91,
    }]


def _fake_index(chunks: list[dict] | None = None,
                errors: list[dict] | None = None) -> dict:
    return {
        "model_name": "mock-model",
        "chunks": chunks if chunks is not None else _fake_hits(),
        "encoder": object(),
        "errors": errors or [],
        "embedded": False,
    }


def test_sanitize_filename_strips_directories_and_specials():
    assert sanitize_filename(r"..\..\evil notes (final).md") == "evil_notes_final.md"
    assert sanitize_filename("clean.pdf") == "clean.pdf"


def test_sanitize_filename_preserves_supported_suffix():
    assert sanitize_filename("\u65e5\u672c\u8a9e.pdf") == "file.pdf"
    long_name = sanitize_filename("a" * 200 + ".md")
    assert long_name.endswith(".md") and len(long_name) <= 120


def test_sanitize_filename_handles_windows_reserved_names():
    assert sanitize_filename("CON.pdf") == "_CON.pdf"


def test_sanitize_filename_rejects_only_dots():
    with pytest.raises(HTTPException):
        sanitize_filename("....")


def test_status_with_empty_corpus_is_zero(client):
    s = client.get("/api/status").json()
    assert s["documents"] == 0
    assert s["uploads"] == []
    assert s["sources"] == []


def test_upload_rejects_unsupported_extension(client):
    res = client.post("/api/upload", files={"file": ("notes.exe", b"MZ", "application/octet-stream")})
    assert res.status_code == 400


def test_upload_sanitizes_name_and_is_listed(client):
    res = client.post("/api/upload", files={"file": ("../../my notes.txt", b"hello corpus", "text/plain")})
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == "my_notes.txt" and body["bytes"] == len(b"hello corpus")
    s = client.get("/api/status").json()
    assert "uploads/my_notes.txt" in s["sources"]
    assert "my_notes.txt" in [f["name"] for f in s["uploads"]]
    assert s["uploads"][0]["origin"] == "uploads/my_notes.txt"


def test_upload_empty_file_rejected(client):
    res = client.post("/api/upload", files={"file": ("empty.txt", b"", "text/plain")})
    assert res.status_code == 400


def test_failed_replacement_keeps_the_original(client):
    first = client.post("/api/upload", files={"file": ("notes.txt", b"original", "text/plain")})
    assert first.status_code == 200

    oversized = b"x" * (1024 * 1024 + 1)
    second = client.post("/api/upload", files={"file": ("notes.txt", oversized, "text/plain")})
    assert second.status_code == 413

    stored = Path(client.app.state.upload_dir) / "notes.txt"
    assert stored.read_bytes() == b"original"


def test_upload_preserves_suffix_for_non_ascii_names(client):
    res = client.post("/api/upload", files={"file": ("\u65e5\u672c\u8a9e.pdf", b"%PDF", "application/pdf")})
    assert res.status_code == 200
    assert res.json()["name"] == "file.pdf"


def test_search_accepts_real_world_source_names(client):
    # The old regex rejected parentheses; validation must now happen against
    # the corpus, not against a character allowlist.
    res = client.post("/api/search", json={"question": "anything at all", "source": "notes (1).md"})
    assert res.status_code == 409  # corpus empty, not a 422 validation error


def test_search_on_empty_corpus_returns_409(client):
    res = client.post("/api/search", json={"question": "anything at all", "top_k": 3})
    assert res.status_code == 409


def test_delete_missing_upload_returns_404(client):
    assert client.delete("/api/uploads/nope.txt").status_code == 404


def test_answer_requires_consent(client):
    res = client.post("/api/answer", json={"question": "hello there", "top_k": 3, "consent": False})
    assert res.status_code == 403


def test_answer_empty_corpus_returns_409(client):
    res = client.post("/api/answer", json={"question": "hello there", "top_k": 3, "consent": True})
    assert res.status_code == 409


def test_answer_success_with_mocked_provider(monkeypatch, client):
    client.post("/api/upload",
                files={"file": ("notes.txt", b"copper vault Oslo archives answer text", "text/plain")})

    fake_hits = _fake_hits()
    monkeypatch.setattr("app.main.srch.create_index", lambda *a, **kw: _fake_index(fake_hits))
    monkeypatch.setattr("app.main.srch.query_index", lambda *a, **kw: fake_hits)
    monkeypatch.setattr(
        "app.main.generate.answer",
        lambda question, hits, model=None: {
            "answer": "Oslo keeps it in a copper vault. [1]",
            "model": "mock-model",
            "sent": [{"rank": 1, "source": "notes.txt", "page": 1, "score": 0.91, "chars": 38}],
            "used": [{**fake_hits[0], "rank": 1}],
            "refused": False,
            "invalid_citations": [],
        },
    )
    res = client.post("/api/answer", json={"question": "where is the vault", "top_k": 1, "consent": True})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "Oslo keeps it in a copper vault. [1]"
    assert body["sent"][0]["source"] == "notes.txt"
    assert body["documents"] == 1
    assert body["hits"][0]["source"] == "notes.txt"
    assert body["hits"][0]["rank"] == 1


def test_answer_only_returns_passages_that_were_sent(monkeypatch, client):
    client.post("/api/upload", files={"file": ("notes.txt", b"alpha beta gamma", "text/plain")})
    hits = [
        {**_fake_hits()[0], "chunk_id": "c1", "rank": None},
        {**_fake_hits()[0], "chunk_id": "c2", "text": "second passage"},
    ]
    hits[0].pop("rank")
    monkeypatch.setattr("app.main.srch.create_index", lambda *a, **kw: _fake_index(hits))
    monkeypatch.setattr("app.main.srch.query_index", lambda *a, **kw: hits)
    monkeypatch.setattr(
        "app.main.generate.answer",
        lambda question, query_hits, model=None: {
            "answer": "Only the first. [1]",
            "model": "mock-model",
            "sent": [{"rank": 1, "source": "notes.txt", "page": 1, "score": 0.91, "chars": 12}],
            "used": [{**hits[0], "rank": 1}],
            "refused": False,
            "invalid_citations": [],
        },
    )
    body = client.post("/api/answer", json={"question": "where", "top_k": 2, "consent": True}).json()
    assert [h["chunk_id"] for h in body["hits"]] == ["c1"]


def test_answer_with_no_hits_refuses_without_calling_groq(monkeypatch, client):
    client.post("/api/upload", files={"file": ("notes.txt", b"alpha beta gamma", "text/plain")})
    monkeypatch.setattr("app.main.srch.create_index", lambda *a, **kw: _fake_index(_fake_hits()))
    monkeypatch.setattr("app.main.srch.query_index", lambda *a, **kw: [])
    # No key: if the endpoint tried to call Groq this would raise 503.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    body = client.post("/api/answer", json={"question": "where", "consent": True}).json()
    assert body["refused"] is True
    assert body["hits"] == []


def test_qdrant_outage_is_reported_as_503(monkeypatch, client):
    client.post("/api/upload", files={"file": ("notes.txt", b"alpha beta gamma", "text/plain")})

    def boom(*args, **kwargs):
        raise StoreUnavailable("Qdrant is not reachable at http://127.0.0.1:6333")

    monkeypatch.setattr("app.main.srch.create_index", boom)
    res = client.post("/api/search", json={"question": "hello there"})
    assert res.status_code == 503
    assert "Qdrant is not reachable" in res.json()["detail"]


def test_index_errors_are_returned_as_warnings(monkeypatch, client):
    client.post("/api/upload", files={"file": ("notes.txt", b"alpha beta gamma", "text/plain")})
    errors = [{"file": "data/broken.pdf", "error": "RuntimeError: bad PDF"}]
    monkeypatch.setattr("app.main.srch.create_index", lambda *a, **kw: _fake_index(errors=errors))
    monkeypatch.setattr("app.main.srch.query_index", lambda *a, **kw: _fake_hits())
    body = client.post("/api/search", json={"question": "hello there"}).json()
    assert body["warnings"] == errors
