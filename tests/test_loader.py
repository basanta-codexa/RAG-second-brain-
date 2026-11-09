from pathlib import Path

import pytest

from loader import load_folder, load_text


def test_load_text_without_markers_is_single_page(tmp_path: Path):
    f = tmp_path / "note.md"
    f.write_text("hello world", encoding="utf-8")
    records = load_text(f)
    assert records == [{"text": "hello world", "source": "note.md", "page": 1}]


def test_load_text_with_markers_and_front_matter(tmp_path: Path):
    f = tmp_path / "note.md"
    f.write_text("FRONT MATTER\n<<<PAGE 2>>>\nbody of page two", encoding="utf-8")
    records = load_text(f)
    pages = [r["page"] for r in records]
    assert 0 in pages  # front matter kept as page 0 instead of dropped
    assert 2 in pages
    by_page = {r["page"]: r for r in records}
    assert "FRONT MATTER" in by_page[0]["text"]
    assert "body of page two" in by_page[2]["text"]


def test_load_folder_rejects_missing_folder(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_folder(str(tmp_path / "nope"))


def test_load_text_falls_back_to_cp1252(tmp_path: Path):
    f = tmp_path / "legacy.txt"
    f.write_bytes(b"caf\xe9 au lait")
    records = load_text(f)
    assert records[0]["text"] == "café au lait"


def test_load_text_rejects_undecodable_bytes(tmp_path: Path):
    f = tmp_path / "binary.txt"
    f.write_bytes(b"\x81\x8d\x90\x9d")
    with pytest.raises(RuntimeError) as exc:
        load_text(f)
    assert "not valid UTF-8 or cp1252" in str(exc.value)
