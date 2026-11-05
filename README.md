# Second Brain — local RAG for your own documents

Indexes your PDFs and Markdown into a vector store and returns the passages
most relevant to a question, with file and page provenance. Optional AI answer
generation (Groq) is off by default; when enabled it answers strictly from the
retrieved passages, cites them inline, and refuses when they don't contain the
answer.

Optional: transcribe handwritten study notes (photos of a PDF) to Markdown
using Google's Gemini vision model. Bring your own API key.

## What it does

- PDF (PyMuPDF/pypdf) and `.md`/`.txt` ingestion; page-aware chunking with overlap
- Embeds with `sentence-transformers` (bge-base by default), stored in **Qdrant** (Docker)
- Cosine top-k retrieval via Qdrant; `search.py` for CLI, web UI for interactive use
- **Incremental indexing**: each file is content-hashed; only new/changed files are re-embedded
- Evaluation suite (`evaluate.py`): hit@1 / hit@k / MRR, reports MRR and flags tests whose source file is missing
- Grounded answers via Groq (opt-in per session): `[1]`-style citations, inline,
  partial answers when only part is supported; refusal sentence otherwise
- Optional `--hybrid` (BM25 + vector RRF) and `--rerank` (cross-encoder) — measured, **not** default

## What it does not do

- No hybrid search by default — tests on this corpus showed it performs worse
- No answer generation without your explicit session consent and a Groq key

## Layout

```
loader.py        reads PDFs/MD/TXT into page records
chunker.py       records -> overlapping, sentence-aware chunks (chunk_id = origin/file)
qdrant_store.py  Qdrant connection, incremental per-file index planner, state
search.py        embed (only changed files) -> upsert -> cosine top-k via Qdrant
evaluate.py      hit@1 / hit@k / MRR on your testset
transcribe.py    handwritten-notes PDF -> Markdown (resumable)
testset.py       retrieval questions with expected (source, page) answers
app/             FastAPI service + vanilla HTML/CSS/JS UI
data/            your corpus (gitignored)
uploads/         files you upload through the UI (gitignored)
transcripts/     OCR page cache (gitignored)
.cache/          incremental-index state (qdrant_state.json) (gitignored)
tests/           pytest suite
```

Markdown files can encode pages with `<<<PAGE n>>>` markers. Content before
the first marker is kept as page 0 rather than dropped. PDFs supply their own
page numbers.

## Setup

Requires Python 3.11+ (developed on Windows) and **Docker** for Qdrant.

```powershell
# 0. Start Qdrant (one-time; persists data under C:\Users\<you>\.qdrant\storage)
docker run -d --name second-brain-qdrant --restart unless-stopped `
  -p 6333:6333 -p 6334:6334 `
  -v "${env:USERPROFILE}\.qdrant\storage:/qdrant/storage" qdrant/qdrant

# already created it once? just make sure it's running:
#   docker start second-brain-qdrant

python -m venv venv
.\venv\Scripts\Activate.ps1

# python -m pip if the pip launcher fails
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt   # unit tests only

copy .env.example .env     # then fill in keys; .env is gitignored
```

If Qdrant is down or unreachable, every command fails fast with a message
telling you to `docker start second-brain-qdrant`. There is no silent fallback.

Keys: `GEMINI_API_KEY` (handwriting OCR), `GROQ_API_KEY` (AI answers, optional).
`transcribe.py` and the web bench read `.env` automatically; an existing
environment variable takes precedence.

## Usage

```powershell
# search
python search.py                        # interactive
python search.py --question "How does the fallback work when the cache is down?"

# inspect what the chunker produced
python chunker.py
python inspect_chunks.py find "cap theorem"

# transcribe handwritten notes (resumable)
python transcribe.py --pdf system-design-notes.pdf

# evaluate retrieval quality
python evaluate.py --k 3
```

Useful flags for evaluation: `--hybrid` (BM25+vector fusion), `--rerank`
(cross-encoder). Example output on this corpus (12 of 45 tests skipped, source
absent): `hit@1: 32/33   hit@3: 33/33   MRR: 0.985`.

Unit tests: `python -m pytest -q`

## Web UI

```powershell
python -m uvicorn app.main:app --port 8000
# http://127.0.0.1:8000
```

1. Drop in a `.pdf`, `.md` or `.txt` (25 MB max). One file at a time; same name replaces.
2. The index syncs on the next query — only changed files are re-embedded
   (first run embeds the corpus once; the UI shows live per-file progress and
   an elapsed timer during embedding).
3. Ask questions; results show source, page, similarity score, and the chunk text.

UI highlights:

- **Source filter** — restrict retrieval to one file (Qdrant payload filter)
- **Advanced retrieval** — Vector / Hybrid / Re-rank, with all scores shown
- **Dark mode** — top-right toggle (or `prefers-color-scheme` on first visit)
- **Keyboard** — `/` focuses the question box; `Ctrl+Enter` submits
- **Copy answer / Copy citations** — one click, citations kept with the answer
- **Live indexing progress** — real per-file/per-chunk counts from the server,
  not a fake timer

AI answer toggle: **Allow AI answers this session**. When on, clicking
**Grounded answer** sends only the retrieved passages to `api.groq.com` with
your key. The UI shows exactly what was sent, before and after. Default model:
`openai/gpt-oss-120b`; override with `GROQ_MODEL`. (Groq's Llama 3.3 70B is
Enterprise-only — 404 on free-tier keys.) The model may use simple arithmetic
on passage numbers and today's date (e.g. age from a birth date) but never
outside knowledge; clicking `[1]` in the answer scrolls to and flashes the
passage it cites.

Endpoints: `GET /api/status`, `GET /api/index-status`, `POST /api/upload`,
`DELETE /api/uploads/<name>`, `POST /api/uploads/clear`, `POST /api/search`,
`POST /api/answer` (`"consent": true` required, enforced server-side).
Search and answer both accept `mode: "vector" | "hybrid" | "rerank"` and
`source: "<file.md>"`.

The UI is a local tool, not a hosted product — don't expose it to the network.

## Corpus privacy

The repository ships no documents. Everything you put in `data/`,
`transcripts/`, `uploads/` and `.cache/` is gitignored and stays local, except:
`transcribe.py` uploads page images to Google, and AI answers upload the
retrieved passages to Groq. Read their free-tier terms before putting private
material through either.

## Measured ranking comparison (this corpus, n=33)

BM25+vector fusion: 30/33 / 32/33 / 0.934 — worse than vector alone.
Cross-encoder re-rank: 28/33 / 32/33 / 0.899 — worse still.
Default remains pure vector (`--hybrid`, `--rerank` are flags). Different
corpora may differ; rerun `evaluate.py` against your own to see.

## Known limits

1. The Qdrant collection and `.cache/qdrant_state.json` are the index. If the
   container or its storage is removed, the first run rebuilds the whole index
   (~1–2 min on this corpus). Keep `~\.qdrant\storage` backed up if your notes matter.
2. Character-based chunking (nearest `. ` boundary). PDF layout, tables and
   math can extract poorly; handwritten PDFs need `transcribe.py` first.
3. AI answers are expendable, not authoritative. A missing passage means
   either an incomplete answer — check the citations.
4. No license. All rights reserved.

## Roadmap

- CI: pytest + lint on GitHub Actions
- Better table and math extraction
