# Contributing

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt
python -m pip install -e . --no-deps
copy .env.example .env
docker compose up -d
```

Tests need no Docker, models, or API keys. Model and SDK imports are lazy, and
the suite replaces Qdrant and the encoder with in-memory fakes.

## Checks

```powershell
python -m pytest
python -m ruff check .
python -m compileall -q chunker.py loader.py search.py qdrant_store.py evaluate.py testset.py transcribe.py inspect_chunks.py app
```

CI runs the same commands on Python 3.11, 3.12, and 3.13. Shared packages are
pinned to identical versions in `requirements.txt` and `requirements-test.txt`;
update both when bumping one.

## Invariants

These are load-bearing. Changes that break them will fail tests or corrupt an index.

- `qdrant_store.plan()` stays a pure function. It decides what to embed and
  delete; `Store` performs the I/O.
- Old point IDs are deleted only after replacement points are upserted. Never
  move that delete ahead of the upsert.
- State is written with a temp file and `os.replace`. Do not write directly to
  the state path.
- A failed file keeps its old points and state entry so the next sync retries.
- Model and SDK imports stay inside the functions that use them. Top-level
  imports of `sentence_transformers`, `google.genai`, or `pymupdf` would make
  the test suite require them.
- `state_lock` in `app/main.py` covers every operation that changes corpus files
  or the cached index, including uploads, deletes, and clears.
- Retrieval changes include a re-run of all four modes and an update to
  `docs/retrieval-evaluation.md`. Comparative claims need a recorded result.

## Adding a retrieval mode

Change these together:

1. `search.py` — candidate generation and `query_index` dispatch.
2. `app/main.py` — the `Mode` literal and flag routing in `run_search`.
3. `app/static/index.html` — the radio option.
4. `evaluate.py` — a CLI flag so the mode can be measured.
5. `tests/` — ranking tests and an API routing test.

## Style

Use US spelling, keep comments to non-obvious decisions, and prefer a short
docstring that says why over one that restates the function name.

## Privacy boundary

Whole documents never leave the machine automatically. First-run model downloads
come from Hugging Face; OCR and answer generation require an explicit opt-in and
send only page images or retrieved passages. Do not add a path that uploads a
full document without a switch in the UI.
