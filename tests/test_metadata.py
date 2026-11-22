from metadata import (
    load_collections,
    load_tags,
    normalize_tags,
    save_collections,
    save_tags,
)


def test_normalize_tags_deduplicates_and_rejects_bad_values():
    assert normalize_tags(["  Roof ", "roof", "Project One", "", "bad!tag", 5]) == [
        "roof", "project one",
    ]


def test_tags_round_trip_to_disk(tmp_path):
    path = tmp_path / "tags.json"
    save_tags({"data/a.md": ["roof", "architecture"]}, path)
    assert load_tags(path) == {"data/a.md": ["roof", "architecture"]}


def test_collections_round_trip_to_disk(tmp_path):
    path = tmp_path / "collections.json"
    save_collections({"data/a.md": "work"}, path)
    assert load_collections(path) == {"data/a.md": "work"}


def test_corrupt_tag_file_is_ignored(tmp_path):
    path = tmp_path / "tags.json"
    path.write_text("not json", encoding="utf-8")
    assert load_tags(path) == {}
