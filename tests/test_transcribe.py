"""Unit tests for the transcription cache and retry state machine.

The Gemini SDK and PyMuPDF are imported lazily by transcribe.py, so these tests
need neither dependency installed.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import transcribe


class ApiError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"api error {code}")
        self.status_code = code


class FakeModels:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


def test_cache_file_uses_zero_padded_numbers():
    assert transcribe.cache_file(Path("cache"), 7) == Path("cache") / "page_007.md"


def test_sha256_file_is_stable(tmp_path: Path):
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    a.write_bytes(b"pdf bytes")
    b.write_bytes(b"pdf bytes")
    assert transcribe.sha256_file(a) == transcribe.sha256_file(b)
    b.write_bytes(b"different")
    assert transcribe.sha256_file(a) != transcribe.sha256_file(b)


def test_bind_cache_writes_manifest(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    pdf = tmp_path / "notes.pdf"
    pdf.write_bytes(b"pdf")
    transcribe.bind_cache_to_pdf(cache, pdf, 12, force=False)
    manifest = transcribe.source_manifest(cache)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data == {"name": "notes.pdf", "sha256": transcribe.sha256_file(pdf), "pages": 12}


def test_bind_cache_rejects_a_different_pdf_unless_forced(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    transcribe.bind_cache_to_pdf(cache, first, 2, force=False)
    with pytest.raises(RuntimeError, match="different PDF"):
        transcribe.bind_cache_to_pdf(cache, second, 2, force=False)
    transcribe.bind_cache_to_pdf(cache, second, 2, force=True)


def test_permanent_failure_classification():
    assert transcribe._permanent_failure(ApiError(401))
    assert transcribe._permanent_failure(ApiError(404))
    assert not transcribe._permanent_failure(ApiError(429))
    assert not transcribe._permanent_failure(ApiError(500))
    assert not transcribe._permanent_failure(ValueError("no code"))


def test_cached_pages_skips_empty_and_unreadable_pages(tmp_path: Path):
    (tmp_path / "page_001.md").write_text("  page one  ", encoding="utf-8")
    (tmp_path / "page_002.md").write_text("   ", encoding="utf-8")
    done, missing = transcribe.cached_pages(tmp_path, 3)
    assert done == {1: "page one"}
    assert missing == [2, 3]


def test_atomic_write_replaces_content(tmp_path: Path):
    target = tmp_path / "page.md"
    target.write_text("old", encoding="utf-8")
    transcribe.atomic_write(target, "new")
    assert target.read_text(encoding="utf-8") == "new"
    assert not (tmp_path / "page.md.tmp").exists()


def test_transcribe_page_retries_transient_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(transcribe, "_image_part", lambda png: "image")
    slept = []
    monkeypatch.setattr(transcribe.time, "sleep", lambda seconds: slept.append(seconds))

    models = FakeModels([ApiError(500), "transcribed"])
    client = SimpleNamespace(models=models)
    result = transcribe.transcribe_page(client, "model", b"png", max_retries=3)
    assert result == "transcribed"
    assert models.calls == 2
    assert len(slept) == 1


def test_transcribe_page_aborts_on_permanent_failure(monkeypatch):
    monkeypatch.setattr(transcribe, "_image_part", lambda png: "image")
    models = FakeModels([ApiError(401)])
    client = SimpleNamespace(models=models)
    with pytest.raises(transcribe.PermanentTranscriptionError):
        transcribe.transcribe_page(client, "model", b"png", max_retries=4)
    assert models.calls == 1  # no retries after a bad key/model


def test_transcribe_page_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(transcribe, "_image_part", lambda png: "image")
    monkeypatch.setattr(transcribe.time, "sleep", lambda seconds: None)
    models = FakeModels([ApiError(500), ApiError(500), ApiError(500)])
    client = SimpleNamespace(models=models)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        transcribe.transcribe_page(client, "model", b"png", max_retries=3)
    assert models.calls == 3
