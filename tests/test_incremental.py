from qdrant_store import plan

MODEL = "test-model"


def make_state(entries, model=MODEL):
    return {"model": model, "collection_built": True,
            "files": {k: {"sha256": h, "point_ids": [f"id-{k}-{i}" for i in range(2)]}
                      for k, h in entries.items()}}


def test_unchanged_corpus_embeds_nothing():
    files = {"data/a.md": "h1", "data/b.md": "h2"}
    out = plan(files, make_state(files), MODEL, collection_exists=True)
    assert out["to_embed"] == []
    assert out["drop_ids"] == []
    assert out["rebuild"] is False


def test_new_file_is_embedded():
    files = {"data/a.md": "h1", "data/b.md": "h2"}
    out = plan(files, make_state({"data/a.md": "h1"}), MODEL, collection_exists=True)
    assert out["to_embed"] == ["data/b.md"]
    assert out["drop_ids"] == []


def test_changed_file_reembeds_and_drops_old_points():
    files = {"data/a.md": "NEW"}
    out = plan(files, make_state({"data/a.md": "OLD"}), MODEL, collection_exists=True)
    assert out["to_embed"] == ["data/a.md"]
    assert out["drop_ids"] == ["id-data/a.md-0", "id-data/a.md-1"]


def test_deleted_file_points_are_dropped():
    files = {"data/a.md": "h1"}
    out = plan(files, make_state({"data/a.md": "h1", "data/gone.md": "h3"}),
               MODEL, collection_exists=True)
    assert out["to_embed"] == []
    assert out["drop_ids"] == ["id-data/gone.md-0", "id-data/gone.md-1"]


def test_model_change_forces_full_rebuild():
    files = {"data/a.md": "h1"}
    out = plan(files, make_state(files, model="old-model"),
               MODEL, collection_exists=True)
    assert out["rebuild"] is True
    assert out["to_embed"] == ["data/a.md"]


def test_missing_collection_forces_full_rebuild():
    files = {"data/a.md": "h1"}
    out = plan(files, make_state(files), MODEL, collection_exists=False)
    assert out["rebuild"] is True
    assert out["to_embed"] == ["data/a.md"]


def test_no_state_embeds_everything():
    files = {"data/a.md": "h1", "uploads/b.md": "h2"}
    out = plan(files, None, MODEL, collection_exists=False)
    assert out["rebuild"] is True
    assert out["to_embed"] == ["data/a.md", "uploads/b.md"]
