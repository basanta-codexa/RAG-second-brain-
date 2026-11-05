from pathlib import Path

from loader import load_text, load_folder


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
    try:
        load_folder(str(tmp_path / "nope"))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
