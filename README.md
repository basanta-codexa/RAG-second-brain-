# Second Brain

A local retrieval-augmented generation (RAG) tool for personal documents. It
indexes PDFs, Markdown, and text files in Qdrant, retrieves relevant passages,
and can generate cited answers with Groq.

It also includes an optional Gemini-based OCR script for handwritten PDFs.

## Requirements

- Python 3.11+
- Docker, for Qdrant
- Optional: `GROQ_API_KEY` for generated answers
- Optional: `GEMINI_API_KEY` for PDF transcription

## Setup

Start Qdrant:

```powershell
docker run -d --name second-brain-qdrant --restart unless-stopped `
  -p 6333:6333 -p 6334:6334 `
  -v "${env:USERPROFILE}\.qdrant\storage:/qdrant/storage" qdrant/qdrant
```

If the container already exists:

```powershell
docker start second-brain-qdrant
```

Create the environment and install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt
copy .env.example .env
```

Fill in the keys you need in `.env`. The web app and `transcribe.py` load that
file automatically.

## Usage

Search the corpus:

```powershell
python search.py
python search.py --question "How does the cache fallback work?"
python search.py --rebuild
```

Inspect chunks:

```powershell
python chunker.py
python inspect_chunks.py find "cap theorem"
```

Evaluate retrieval:

```powershell
python evaluate.py --k 3
python evaluate.py --k 3 --hybrid
python evaluate.py --k 3 --rerank
python evaluate.py --k 3 --fail-under 0.8
```

`evaluate.py` reports hit@1, hit@k, and MRR. It also reports strict and lenient
subsets: strict tests require the correct page in a multi-page source; lenient
tests accept any page or come from a single-page source.

Transcribe a handwritten PDF:

```powershell
python transcribe.py --pdf system-design-notes.pdf
```

The transcription is resumable. Pages are cached under `transcripts/`.

Run the web app:

```powershell
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`. The app accepts `.pdf`, `.md`, and `.txt` uploads
up to 25 MB by default. The index updates on the next query.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `SECOND_BRAIN_DATA` | `data` | Base corpus directory |
| `SECOND_BRAIN_UPLOADS` | `uploads` | Web upload directory |
| `SECOND_BRAIN_MODEL` | `BAAI/bge-base-en-v1.5` | Embedding model |
| `SECOND_BRAIN_MAX_MB` | `25` | Maximum upload size |
| `SECOND_BRAIN_CHUNK_SIZE` | `800` | Chunk size in characters |
| `SECOND_BRAIN_OVERLAP` | `150` | Chunk overlap in characters |
| `SECOND_BRAIN_QDRANT` | `http://127.0.0.1:6333` | Qdrant URL |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Answer model |

## How it works

```text
loader.py        PDF/Markdown/text -> page records
chunker.py       page records -> overlapping, sentence-aware chunks
qdrant_store.py  collection lifecycle, filters, and index state
search.py        incremental embedding, retrieval, hybrid search, reranking
evaluate.py      retrieval metrics against testset.py
transcribe.py    handwritten PDF -> Markdown; resumable page cache
app/             FastAPI service and browser UI
data/            base corpus (gitignored)
uploads/         web uploads (gitignored)
transcripts/     OCR cache (gitignored)
.cache/          index state and progress (gitignored)
tests/           pytest suite
```

Each file is content-hashed. Only new or changed files are embedded. Qdrant
state is stored in `.cache/qdrant_state.json`; a change to the embedding model,
chunk configuration, or collection layout triggers a full rebuild.

Retrieval defaults to vector search. Hybrid BM25/vector search and cross-encoder
reranking are available through the CLI and web UI. Source filters accept an
origin such as `data/notes.md` or a filename such as `notes.pdf`.

The web API exposes:

- `GET /api/status`
- `GET /api/index-status`
- `POST /api/upload`
- `DELETE /api/uploads/<name>`
- `POST /api/uploads/clear`
- `POST /api/search`
- `POST /api/answer`

## Privacy and security

Documents stay local. The two exceptions are explicit:

- `transcribe.py` sends page images to Google Gemini.
- Generated answers send retrieved passages to Groq after you enable AI answers
  for the session.

The web app has no authentication and is intended for local use only. Do not
expose it to a network. Generated answers are not verified for factual
correctness; treat them as drafts.

## Tests

```powershell
python -m pytest
python -m ruff check .
```

The test suite uses an in-memory Qdrant client and a fake encoder, so it does
not need Docker or a downloaded model.

## Limitations

- PDF extraction quality depends on the source layout. Tables, math, and scans
  may need OCR or manual cleanup.
- The web app caches the index by file size and modification time. After
  restoring files with preserved timestamps, restart the server or run
  `python search.py --rebuild`.
- If Qdrant storage is removed while the web app is running, restart the app so
  its index cache notices the missing collection.
- Markdown page numbers require `<<<PAGE n>>>` markers. Without them, the file
  is treated as page 1 for evaluation.
- No license is provided.
