"""Shared test isolation.

Tests must not write to the repository's ``.cache`` directory. A stale or
locked progress file there can make an unrelated test fail, and a test run
should never disturb a developer's live index.
"""
import pytest

import search


@pytest.fixture(autouse=True)
def isolated_index_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "PROGRESS_PATH", tmp_path / "index_progress.json")
