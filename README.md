# Second Brain — grounded local RAG for your own documents

A small, honest **retrieval** pipeline: it indexes your own PDFs / Markdown / text
into an embedding vector space and returns the most relevant passages for a
question, with file and page provenance. Answer generation is **opt-in**: with
your Groq key and a per-session consent toggle, it writes answers strictly
grounded in the retrieved passages, cites them inline, and refuses when the
documents don't say. The default experience is evidence-first.

Optional: transcribe handwritten study notes (photos of a PDF) into clean
Markdown using Google's Gemini vision model, bringing your own API key.

## Status: honest scope

| Works today | Not implemented yet |
| --- | --- |
| PDF (PyMuPDF/pypdf) and `.md`/`.txt` ingestion | Vector database / ANN at scale |
| Page-aware chunking with overlap | Hybrid keyword+vector retrieval |
| Cosine top-k retrieval with sentence-transformers (bge-base by default) | Incremental per-file re-indexing |
| Persistent embedding cache (`.cache/`, content-hash keyed) | Any form of cloud upload/sync of your notes |
| Optional handwritten-notes transcription via Gemini | — |
| Honest evaluation: hit@1, hit@k, MRR, missing-source skips | — |
| Grounded AI answers via Groq — session consent, `[1]`-style citations, refuses when the documents don't say | — |

## How it fits together

```
data/*.pdf,*.md,*.txt
   │
   ▼ loader.py      → page-level records ({text, source, page})
   ▼ chunker.py     → overlapping, sentence-aware chunks with chunk_id
   ▼ search.py      → embed (BGE) → cache → cosine top-k
   ▼ evaluate.py    → hit@1 / hit@k / MRR on your labelled test questions
```

Markdown files may encode pages with a `<<<PAGE n>>>` marker; content before
the first marker is kept as page 0 (never silently dropped). PDFs contribute
their real page numbers.

```
app/
├── main.py      # FastAPI service: upload, clear uploads, merged search
└── static/      # vanilla HTML/CSS/JS single page — no frameworks
uploads/         # gitignored staging area for uploaded files
```

## Bring your own data

The repository intentionally ships **no documents** — your notes stay private.
Put *your* files in `data/`:

```
second-brain/
└── data/            # gitignored
    ├── notes.pdf
    ├── playbook.md
    └── book_notes.txt
```

Then search, browse chunks or evaluate exactly as below. `data/`, `transcripts/`
and `system-design-notes.pdf` are gitignored so personal or copyrighted material
is never pushed.

## Setup

Requires Python 3.11+ (developed on Windows; paths are portable).

```powershell
# 1. Create and activate a virtual environment (kept OUTSIDE OneDrive sync!)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies (use python -m pip if the launcher fails)
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt   # needed only for the unit tests

# 3. (Optional) set your API keys
copy .env.example .env                # then edit .env — never commit it
```

`transcribe.py` and the web bench load keys from the project `.env` file automatically
(`GEMINI_API_KEY` for handwriting OCR, `GROQ_API_KEY` for grounded answers).
An already-set environment variable takes precedence.

## Usage

```powershell
# Interactive search
python search.py
# one-shot question
python search.py --question "How does the fallback work when the cache is down?"
# force re-embedding (default reuses .cache/ if corpus is unchanged)
python search.py --rebuild

# Preview chunking / inspect chunks
python chunker.py
python inspect_chunks.py find "cap theorem"

# Transcribe handwritten notes (resumable)
python transcribe.py --pdf system-design-notes.pdf

# Evaluate retrieval quality
python evaluate.py --k 3
```

Notes on the commands above:

- `search.py` auto-applies the BGE query instruction prefix only when the model
  name contains `bge`; override with `--no-query-prefix`.
- `evaluate.py` **skips** questions whose ground-truth file is missing (reported,
  never silently counted), flags *lenient* answers (any page accepted), and
  reports **hit@1, hit@k and MRR**. Example of a run with 12 of 45 questions
  skipped because that source was not present: `hit@1: 32/33 hit@3: 33/33 MRR: 0.985`.
- Unit tests: `python -m pytest -q`

## Web test bench

A minimal, local web UI ships with the repo — no frameworks, no dashboards,
no account, no telemetry. It lets you upload a document, then ask questions and
inspect the exact chunks that come back (file, page, similarity score). The
merged corpus is `data/` plus anything you upload; uploads stay in the
gitignored `uploads/` folder.

```powershell
# Start the server (loads PyTorch; first start takes ~10s)
python -m uvicorn app.main:app --port 8000

# Open http://127.0.0.1:8000
```

What you get, end to end:

1. **Upload** any `.pdf`, `.md` or `.txt` (25 MB max, sanitised file name,
   empty payloads rejected).
2. The **index rebuilds automatically** on the next search — on CPU this
   typically takes 1–2 minutes; afterwards it is reused from the content-hash
   cache under `.cache/`.
3. **Ask** a question (3–2000 characters, 1–10 passages) and read the evidence —
   scores, source file, page, and the raw chunk text with expand/collapse.

**Grounded AI answers (opt-in, via Groq)**

- Flip **Allow AI answers this session** — you'll be asked to confirm exactly
  what that means (retrieved passages are sent to `api.groq.com` using your key;
  full documents are never sent). The setting expires when the page reloads.
- Click **Grounded answer**. The model must answer *only* from the retrieved
  passages, cite `[1]`, `[2]` inline, and otherwise reply
  `I don't know — the documents don't answer this.`
- Every answer ships with an audit line: what was sent, how big it was, which
  model. Expand **What was sent to Groq** to inspect it.
- Requires `GROQ_API_KEY` in `.env` (free tier at console.groq.com/keys).
  Without it, generation returns a clear error and the rest of the UI is unaffected.
  Override the model with `GROQ_MODEL` if your account offers different models.

Endpoints if you prefer the API: `GET /api/status`, `POST /api/upload`,
`DELETE /api/uploads/<name>`, `POST /api/uploads/clear`, `POST /api/search`,
`POST /api/answer` (requires `"consent": true` in the body — enforced server-side).
Configuration via `SECOND_BRAIN_DATA`, `SECOND_BRAIN_UPLOADS`,
`SECOND_BRAIN_MODEL`, `SECOND_BRAIN_MAX_MB`, `GROQ_MODEL`.

The UI is a **local test harness**, not a hosted product — do not expose it
on the network; the upload endpoint trusts your local filesystem boundaries.

## Honest limitations (no sugarcoating)

1. **AI answers exist but are opt-in and expendable.** Retrieval is the core;
   the Groq answer is a layer on top. If retrieval misses, generation will
   refuse (by design) rather than invent — but the quality of that refusal
   still depends on the top-k you picked.
2. **Only retrieved passages leave your machine** (when you enable AI answers),
   and only with the session consent toggle on. Your whole library is never
   bulk-uploaded. You still must read Groq's and Google's free-tier terms
   before putting anything sensitive through them.
2. **Re-embeds when content changes.** The cache fingerprints file contents —
   any edit re-embeds the whole corpus. Fine at personal scale; use an ANN index
   for large corpora.
3. **Character-based chunking** and sentence-aware only up to nearest `. `; PDF
   layout, tables and math may extract poorly.
4. **Two embedding backends tested**: `all-MiniLM-L6-v2` and `BAAI/bge-*`; BGE is
   the default. Model is user-selectable (`--model`).
5. **Your data is your responsibility**: do not commit copyrighted books or PDFs;
   the Gemini transcription step uploads page images to Google — read their terms
   for the free tier before transcribing anything sensitive.

## Privacy and secrets

- `.env` is gitignored and **never** committed. `transcribe.py` and the web
  bench load keys from this file; `.env.example` ships as a placeholder.
- `data/`, `transcripts/`, `uploads/`, `*.pdf` at the repo root, and `.cache/`
  are ignored.
- The repo ships no documents at all, by design.

## License

No license is granted. All rights reserved. The code may not be reused without
permission.

## Roadmap

- FAISS or sqlite-vec backed persistent vector index
- BM25 + vector hybrid retrieval; cross-encoder reranking
- GitHub Actions CI (pytest + lint)
- OCR-free page anchors; better table handling
