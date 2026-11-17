# Second Brain

Local retrieval over personal documents. Index PDFs, Markdown, and text into
Qdrant, search them from a CLI or browser, and optionally generate answers that
cite the passages they used. A separate script transcribes handwritten PDFs
through Gemini before indexing.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-89%20passing-0A9EDC?logo=pytest&logoColor=white)](tests/)

## Quick start

Requires Python 3.11+ and Docker.

```powershell
docker compose up -d
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt
python -m pip install -e . --no-deps
copy .env.example .env
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` and upload a `.pdf`, `.md`, or `.txt` file. The app
accepts files up to 25 MB. Qdrant starts with telemetry disabled; see
[Privacy](#privacy-and-security) for the other network calls.

Command line:

```powershell
python search.py --question "How does the cache fallback work?"
python search.py --rebuild
python evaluate.py --k 3
```

## Interface

The workbench shows the corpus and the question form. Search returns ranked
passages with source and page. No answer is generated unless the consent switch
is enabled.

![Workbench: upload panel and question form](docs/screenshots/01-workbench.png)

Advanced options expose the four retrieval modes described below.

![Advanced retrieval options](docs/screenshots/02-retrieval-modes.png)

Ranked passages for a question about the CAP theorem.

![Search results](docs/screenshots/03-search-results.png)

The same view in dark mode.

![Dark mode](docs/screenshots/04-dark-mode.png)

## How it works

Documents are loaded as page records, split into overlapping sentence-aware
chunks, embedded with `BAAI/bge-base-en-v1.5`, and written to Qdrant. Each file
is hashed; only new or changed files are re-embedded. Deletions of old point IDs
happen after replacements are in place, and the state file is written
atomically.

```mermaid
flowchart LR
    A[PDF / Markdown / TXT] --> B[loader.py]
    B --> C[chunker.py]
    C --> D[search.py<br/>embed changed files]
    D --> E[(Qdrant)]
    Q[Question] --> R[search.py<br/>embed query]
    R --> E
    E --> S[Ranked passages]
    S --> T[CLI]
    S --> U[FastAPI + web UI]
    U -. opt-in .-> V[Groq answer]
    A -. optional OCR .-> W[transcribe.py + Gemini]
    W --> A
```

Details are in [docs/architecture.md](docs/architecture.md).

## Retrieval modes

| Mode | Behavior |
| --- | --- |
| `vector` | Dense cosine search in Qdrant. Default. |
| `hybrid` | BM25 and vector scores fused by reciprocal rank fusion. |
| `rerank` | Cross-encoder reranking of the vector top 20. |
| `hybrid+rerank` | Fusion first, then cross-encoder reranking of the fused pool. |

The default is `vector`. On the 45-case test set it scored 19/22 on strict hit@1
and answered in 47 ms; `hybrid+rerank` scored 15/22 at roughly 1.1 s. The full
comparison, including two reranker models, is in
[docs/retrieval-evaluation.md](docs/retrieval-evaluation.md).

Hybrid modes score against a BM25 index and vector matrix built once per index
sync, so queries do not pull the whole corpus over HTTP. Scoring is still
O(corpus) per query: about 1.5 ms for 3,000 chunks at 768 dimensions.

Source filters take either an origin (`data/notes.md`) or a filename
(`notes.pdf`).

## Evaluation

```powershell
python evaluate.py --k 3
python evaluate.py --k 3 --hybrid
python evaluate.py --k 3 --rerank
python evaluate.py --k 3 --hybrid-rerank
python evaluate.py --k 3 --fail-under 0.8
```

`--fail-under` checks the strict subset when one exists. Strict cases require
the correct page in a multi-page source; lenient cases accept any page from the
source or come from a single-page source.

| Metric | Definition |
| --- | --- |
| `hit@1` | Correct source and page ranked first. |
| `hit@k` | Correct source and page in the top *k*. |
| `MRR` | Mean reciprocal rank of the first correct result. |

Cases whose source file is missing are skipped and printed, not counted as
passes.

## Configuration

Read from the environment or a `.env` file in the project root.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SECOND_BRAIN_DATA` | `data` | Corpus directory |
| `SECOND_BRAIN_UPLOADS` | `uploads` | Upload directory |
| `SECOND_BRAIN_MODEL` | `BAAI/bge-base-en-v1.5` | Embedding model |
| `SECOND_BRAIN_RERANKER` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `SECOND_BRAIN_CHUNK_SIZE` | `800` | Chunk size, characters |
| `SECOND_BRAIN_OVERLAP` | `150` | Chunk overlap, characters |
| `SECOND_BRAIN_MAX_MB` | `25` | Upload limit |
| `SECOND_BRAIN_QDRANT` | `http://127.0.0.1:6333` | Qdrant endpoint |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Answer model |

Changing the embedding model, chunk size, or overlap invalidates the index and
triggers a full rebuild.

## Web API

Interactive documentation is served at `/docs` while the app is running.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/status` | Corpus, sources, model, upload limit |
| `GET` | `/api/index-status` | Indexing progress |
| `POST` | `/api/upload` | Upload one document |
| `DELETE` | `/api/uploads/{name}` | Remove one upload |
| `POST` | `/api/uploads/clear` | Remove all uploads |
| `POST` | `/api/search` | Retrieve passages |
| `POST` | `/api/answer` | Generate an answer (requires consent) |

## Privacy and security

Documents, chunks, and index state stay on this machine. Three exceptions:

- The first run downloads the embedding model, and rerank modes download a
  cross-encoder, from Hugging Face.
- `transcribe.py` sends rendered page images to Gemini when OCR is used.
- `/api/answer` sends retrieved passages to Groq after the UI's consent switch
  is enabled.

`docker-compose.yml` sets `QDRANT__TELEMETRY_DISABLED=true`. The web app has no
authentication and binds to localhost; do not expose it to a network. Generated
answers are not factually verified, and citation checks only catch reference
numbers that were not sent to the model.

## Tests

```powershell
python -m pytest
python -m ruff check .
```

Tests use in-memory Qdrant and fake encoders, so they need neither Docker nor
downloaded models. Development conventions are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Project layout

```text
app/          FastAPI service, UI, answer generation
tests/        pytest suite
docs/         architecture and evaluation results
data/         corpus (gitignored)
uploads/      web uploads (gitignored)
transcripts/  OCR page cache (gitignored)
loader.py     documents -> page records
chunker.py    page records -> chunks
qdrant_store.py  collection lifecycle, index plan, state
search.py     embedding, retrieval, CLI
evaluate.py   retrieval metrics
transcribe.py  handwritten PDF -> Markdown
testset.py    retrieval test cases
```

`pip install -e .` adds `second-brain-search` and `second-brain-evaluate`.

## Limitations

- PDF text extraction varies with layout; tables, equations, and scans may need
  OCR or manual cleanup.
- The web index cache keys on file size and mtime. Restoring files with
  preserved timestamps needs a rebuild.
- Removing Qdrant storage while the app runs requires an app restart.
- Markdown page numbers require `<<<PAGE n>>>` markers; without them the file is
  treated as page 1.
- Only 26 of 39 source pages were transcribed. The evaluation document covers
  every page referenced by the test set.

## License

MIT. See [LICENSE](LICENSE).
