from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app, sanitize_filename


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


def test_sanitize_filename_strips_directories_and_specials():
    assert sanitize_filename(r"..\..\evil notes (final).md") == "evil_notes_final.md"
    assert sanitize_filename("clean.pdf") == "clean.pdf"


def test_sanitize_filename_rejects_only_dots():
    with pytest.raises(Exception):
        sanitize_filename("....")


def test_status_with_empty_corpus_is_zero(client):
    s = client.get("/api/status").json()
    assert s["documents"] == 0
    assert s["uploads"] == []


def test_upload_rejects_unsupported_extension(client):
    res = client.post("/api/upload", files={"file": ("notes.exe", b"MZ", "application/octet-stream")})
    assert res.status_code == 400


def test_upload_sanitizes_name_and_is_listed(client):
    res = client.post("/api/upload", files={"file": ("../../my notes.txt", b"hello corpus", "text/plain")})
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == "my_notes.txt" and body["bytes"] == len(b"hello corpus")
    s = client.get("/api/status").json()
    assert "my_notes.txt" in [f["name"] for f in s["uploads"]]


def test_upload_empty_file_rejected(client):
    res = client.post("/api/upload", files={"file": ("empty.txt", b"", "text/plain")})
    assert res.status_code == 400


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

    fake_hits = [
        {"text": "copper vault Oslo archives answer text", "source": "notes.txt",
         "page": 1, "chunk_id": "notes.txt-p1-c0", "score": 0.91},
    ]
    fake_index = {"model_name": "mock-model", "fingerprint": "x", "encoder": object()}
    monkeypatch.setattr("app.main.srch.create_index", lambda *a, **kw: fake_index)
    monkeypatch.setattr("app.main.srch.query_index", lambda *a, **kw: fake_hits)
    monkeypatch.setattr(
        "app.main.generate.answer",
        lambda question, hits, model=None: {
            "answer": "Oslo keeps it in a copper vault. [1]",
            "model": "mock-model",
            "sent": [{"rank": 1, "source": "notes.txt", "page": 1, "score": 0.91, "chars": 38}],
            "refused": False,
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
