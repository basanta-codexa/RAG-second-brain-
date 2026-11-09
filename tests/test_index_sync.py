"""Exercise create_index's destructive operations without a Qdrant server.

These tests are intentionally about ordering: drop/delete must never run after
the collection is gone, and old points must only be deleted once replacements
have been upserted.
"""
import copy
from pathlib import Path

import numpy as np
import pytest

import search
from qdrant_store import StoreUnavailable, point_id


class FakeEncoder:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        return np.ones((len(texts), 4), dtype=np.float32)


class FakeStore:
    state = None
    points: dict[str, dict] = {}
    events: list = []
    exists_flag = False
    fail_upsert = False
    collection = "fake"

    @classmethod
    def reset(cls):
        cls.state = None
        cls.points = {}
        cls.events = []
        cls.exists_flag = False
        cls.fail_upsert = False

    def __init__(self, *args, **kwargs):
        pass

    def exists(self):
        return type(self).exists_flag

    def drop(self):
        type(self).events.append("drop")
        type(self).points = {}
        type(self).exists_flag = False

    def create(self, vector_size):
        type(self).events.append("create")
        type(self).exists_flag = True

    def load_state(self):
        return copy.deepcopy(type(self).state)

    def save_state(self, state):
        type(self).events.append("save")
        type(self).state = copy.deepcopy(state)

    def delete_points(self, ids):
        if not ids:  # real Store returns early; no-op deletes are not events
            return
        type(self).events.append(("delete", tuple(ids)))
        for pid in ids:
            type(self).points.pop(pid, None)

    def upsert(self, chunks, vectors):
        if type(self).fail_upsert:
            raise StoreUnavailable("Qdrant upsert failed at http://fake")
        ids = [point_id(c["chunk_id"]) for c in chunks]
        type(self).events.append(("upsert", tuple(ids)))
        for c, pid in zip(chunks, ids, strict=True):
            type(self).points[pid] = dict(c)
        return ids

    def scroll_all(self, with_vectors=False, source=None):
        return [dict(payload, id=pid) for pid, payload in type(self).points.items()]

    def count(self):
        return len(type(self).points)


@pytest.fixture(autouse=True)
def fake_store(monkeypatch):
    FakeStore.reset()
    monkeypatch.setattr("search.Store", FakeStore)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(data: Path, model: str = "m1", rebuild: bool = False):
    return search.create_index(str(data), model, chunk_size=60, overlap=10,
                               rebuild=rebuild, encoder=FakeEncoder())


def test_first_index_embeds_and_creates(tmp_path: Path):
    data = tmp_path / "data"
    write(data / "a.md", "alpha beta gamma. " * 20)

    index = run(data)
    assert index["errors"] == []
    assert "create" in FakeStore.events
    assert "drop" in FakeStore.events
    assert not any(isinstance(e, tuple) and e[0] == "delete" for e in FakeStore.events)
    assert FakeStore.state["files"]["data/a.md"]["sha256"]


def test_unchanged_corpus_does_not_reembed(tmp_path: Path):
    data = tmp_path / "data"
    write(data / "a.md", "alpha beta gamma. " * 20)
    run(data)
    FakeStore.events.clear()

    run(data)
    assert FakeStore.events == ["save"]  # sync bookkeeping only


def test_changed_file_upserts_before_deleting_old_points(tmp_path: Path):
    data = tmp_path / "data"
    write(data / "a.md", "alpha beta gamma. " * 40)
    run(data)
    old_ids = set(FakeStore.state["files"]["data/a.md"]["point_ids"])

    write(data / "a.md", "short. text. only.")  # fewer chunks than before
    FakeStore.events.clear()
    index = run(data)

    kinds = [e[0] if isinstance(e, tuple) else e for e in FakeStore.events]
    assert "upsert" in kinds and "delete" in kinds
    assert kinds.index("upsert") < kinds.index("delete")
    deleted = {pid for e in FakeStore.events if isinstance(e, tuple) and e[0] == "delete" for pid in e[1]}
    assert deleted <= old_ids
    assert index["errors"] == []


def test_model_change_drops_and_never_deletes_against_a_missing_collection(tmp_path: Path):
    data = tmp_path / "data"
    write(data / "a.md", "alpha beta gamma. " * 20)
    run(data)
    FakeStore.events.clear()

    run(data, model="m2")
    kinds = [e[0] if isinstance(e, tuple) else e for e in FakeStore.events]
    assert "drop" in kinds
    assert "delete" not in kinds  # a point delete on a dropped collection is a 404


def test_rebuild_flag_drops_and_never_deletes(tmp_path: Path):
    data = tmp_path / "data"
    write(data / "a.md", "alpha beta gamma. " * 20)
    run(data)
    FakeStore.events.clear()

    run(data, rebuild=True)
    kinds = [e[0] if isinstance(e, tuple) else e for e in FakeStore.events]
    assert "drop" in kinds
    assert "delete" not in kinds


def test_broken_changed_file_keeps_old_points_and_state(tmp_path: Path):
    data = tmp_path / "data"
    write(data / "a.md", "alpha beta gamma. " * 20)
    run(data)
    old_state = copy.deepcopy(FakeStore.state)
    old_points = set(FakeStore.points)

    (data / "a.md").write_bytes(b"\x81\x8d\x90\x9d")  # neither UTF-8 nor cp1252
    FakeStore.events.clear()
    index = run(data)

    assert len(index["errors"]) == 1
    assert "a.md" in index["errors"][0]["file"]
    assert set(FakeStore.points) == old_points
    assert FakeStore.state["files"]["data/a.md"] == old_state["files"]["data/a.md"]
    kinds = [e[0] if isinstance(e, tuple) else e for e in FakeStore.events]
    assert "delete" not in kinds


def test_backend_outage_aborts_without_saving_a_half_state(tmp_path: Path):
    data = tmp_path / "data"
    write(data / "a.md", "alpha beta gamma. " * 20)
    run(data)
    old_state = copy.deepcopy(FakeStore.state)
    old_points = set(FakeStore.points)

    write(data / "a.md", "a changed document. " * 20)
    FakeStore.fail_upsert = True
    with pytest.raises(StoreUnavailable):
        run(data)

    assert FakeStore.state == old_state
    assert set(FakeStore.points) == old_points
