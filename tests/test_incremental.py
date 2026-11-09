from qdrant_store import index_signature, plan

MODEL = "test-model"
SIG = index_signature(MODEL, 800, 150)


def make_state(entries, signature=SIG):
    return {"signature": signature,
            "files": {k: {"sha256": h, "point_ids": [f"id-{k}-{i}" for i in range(2)]}
                      for k, h in entries.items()}}


def test_unchanged_corpus_embeds_nothing():
    files = {"data/a.md": "h1", "data/b.md": "h2"}
    out = plan(files, make_state(files), SIG, collection_exists=True)
    assert out["to_embed"] == []
    assert out["drop_by_key"] == {}
    assert out["rebuild"] is False


def test_new_file_is_embedded():
    files = {"data/a.md": "h1", "data/b.md": "h2"}
    out = plan(files, make_state({"data/a.md": "h1"}), SIG, collection_exists=True)
    assert out["to_embed"] == ["data/b.md"]
    assert out["drop_by_key"] == {}


def test_changed_file_reembeds_and_marks_old_points_for_drop():
    files = {"data/a.md": "NEW"}
    out = plan(files, make_state({"data/a.md": "OLD"}), SIG, collection_exists=True)
    assert out["to_embed"] == ["data/a.md"]
    assert out["drop_by_key"] == {"data/a.md": ["id-data/a.md-0", "id-data/a.md-1"]}


def test_deleted_file_points_are_dropped():
    files = {"data/a.md": "h1"}
    out = plan(files, make_state({"data/a.md": "h1", "data/gone.md": "h3"}),
               SIG, collection_exists=True)
    assert out["to_embed"] == []
    assert out["drop_by_key"] == {"data/gone.md": ["id-data/gone.md-0", "id-data/gone.md-1"]}


def test_model_change_forces_full_rebuild_without_drop_ids():
    files = {"data/a.md": "h1"}
    old_sig = index_signature("old-model", 800, 150)
    out = plan(files, make_state(files, signature=old_sig), SIG, collection_exists=True)
    assert out["rebuild"] is True
    assert out["to_embed"] == ["data/a.md"]
    # The collection is dropped, so deleting individual ids would be a 404.
    assert out["drop_by_key"] == {}


def test_chunking_change_forces_full_rebuild():
    files = {"data/a.md": "h1"}
    old_sig = index_signature(MODEL, 400, 50)
    out = plan(files, make_state(files, signature=old_sig), SIG, collection_exists=True)
    assert out["rebuild"] is True
    assert out["to_embed"] == ["data/a.md"]


def test_missing_collection_forces_full_rebuild():
    files = {"data/a.md": "h1"}
    out = plan(files, make_state(files), SIG, collection_exists=False)
    assert out["rebuild"] is True
    assert out["to_embed"] == ["data/a.md"]
    assert out["drop_by_key"] == {}


def test_no_state_embeds_everything():
    files = {"data/a.md": "h1", "uploads/b.md": "h2"}
    out = plan(files, None, SIG, collection_exists=False)
    assert out["rebuild"] is True
    assert out["to_embed"] == ["data/a.md", "uploads/b.md"]


def test_old_state_without_signature_forces_rebuild():
    files = {"data/a.md": "h1"}
    legacy = {"model": MODEL, "files": {"data/a.md": {"sha256": "h1", "point_ids": []}}}
    out = plan(files, legacy, SIG, collection_exists=True)
    assert out["rebuild"] is True


def test_corrupt_state_forces_rebuild():
    files = {"data/a.md": "h1"}
    corrupt = {"signature": SIG, "files": {"data/a.md": {"sha256": "h1"}}}
    out = plan(files, corrupt, SIG, collection_exists=True)
    assert out["rebuild"] is True
