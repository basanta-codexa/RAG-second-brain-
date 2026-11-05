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
- Embeds with `sentence-transformers` (bge-base by default) — cached by content hash under `.cache/`
- Cosine top-k retrieval; `search.py` for CLI, web UI for interactive use
- Evaluation suite (`evaluate.py`): hit@1 / hit@k / MRR, reports MRR and flags tests whose source file is missing
- Grounded answers via Groq (opt-in per session): `[1]`-style citations, inline,
  partial answers when only part is supported; refusal sentence otherwise
- Optional `--hybrid` (BM25 + vector RRF) and `--rerank` (cross-encoder) — measured, **not** default

## What it does not do

- No vector database or ANN index (brute-force dot product; fine at this scale)
- No hybrid search by default — tests on this corpus showed it performs worse
- No answer generation without your explicit session consent and a Groq key

## Layout

```
loader.py      reads PDFs/MD/TXT into page records
chunker.py     records -> overlapping, sentence-aware chunks (with chunk_id)
search.py      embed -> cache -> cosine top-k
evaluate.py    hit@1 / hit@k / MRR on your testset
transcribe.py  handwritten-notes PDF -> Markdown (resumable)
testset.py     retrieval questions with expected (source, page) answers
app/           FastAPI service + vanilla HTML/CSS/JS UI
data/          your corpus (gitignored)
uploads/       files you upload through the UI (gitignored)
transcripts/   OCR page cache (gitignored)
.cache/        embedding cache (gitignored)
tests/         pytest suite
```

Markdown files can encode pages with `<<<PAGE n>>>` markers. Content before
the first marker is kept as page 0 rather than dropped. PDFs supply their own
page numbers.

## Setup

Python 3.11+ (developed on Windows).

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

# python -m pip if the pip launcher fails
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt   # unit tests only

copy .env.example .env     # then fill in keys; .env is gitignored
```

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
2. The index rebuilds on the next query (CPU-bound; ~1–2 min on first run, then cached).
3. Ask questions; results show source, page, similarity score, and the chunk text.

AI answer toggle: **Allow AI answers this session**. When on, clicking
**Grounded answer** sends only the retrieved passages to `api.groq.com` with
your key. The UI shows exactly what was sent, before and after. Default model:
`openai/gpt-oss-120b`; override with `GROQ_MODEL`. (Groq's Llama 3.3 70B is
Enterprise-only — 404 on free-tier keys.)

Endpoints: `GET /api/status`, `POST /api/upload`, `DELETE /api/uploads/<name>`,
`POST /api/uploads/clear`, `POST /api/search`, `POST /api/answer`
(`"consent": true` required, enforced server-side).

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

1. Retrieval is brute-force dot product over ~hundreds of chunks — fine here,
   not a database.
2. Any edit to the corpus re-embeds everything (fingerprint is content hash).
   There is no incremental per-file indexing.
3. Character-based chunking (nearest `. ` boundary). PDF layout, tables and
   math can extract poorly; handwritten PDFs need `transcribe.py` first.
4. AI answers are expendable, not authoritative. A missing passage means either
   an incomplete answer — check the citations.
5. No license. All rights reserved.

## Roadmap

- Persistent ANN index (FAISS or sqlite-vec)
- CI: pytest + lint on GitHub Actions
- Better table and math extraction
