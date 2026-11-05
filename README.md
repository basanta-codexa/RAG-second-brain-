# Second Brain — grounded local RAG for your own documents

A small, honest **retrieval** pipeline: it indexes your own PDFs / Markdown / text
into an embedding vector space and returns the most relevant passages for a
question, with file and page provenance. It does **not** generate answers yet —
by design it shows you the evidence instead of hallucinating one.

Optional: transcribe handwritten study notes (photos of a PDF) into clean
Markdown using Google's Gemini vision model, bringing your own API key.

## Status: honest scope

| Works today | Not implemented yet |
| --- | --- |
| PDF (PyMuPDF/pypdf) and `.md`/`.txt` ingestion | Answer generation / LLM synthesis |
| Page-aware chunking with overlap | Vector database / ANN at scale |
| Cosine top-k retrieval with sentence-transformers (bge-base by default) | Hybrid keyword+vector retrieval |
| Persistent embedding cache (`.cache/`, content-hash keyed) | Incremental per-file re-indexing |
| Optional handwritten-notes transcription via Gemini | Any form of cloud upload/sync of your notes |
| Honest evaluation: hit@1, hit@k, MRR, missing-source skips | — |

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

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt   # needed only for the unit tests

# 3. (Optional, only for transcribe.py) set your Gemini API key
copy .env.example .env                # then edit .env — never commit it
$env:GEMINI_API_KEY = "your_key_here"
```

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

## Honest limitations (no sugarcoating)

1. **It retrieves; it does not answer.** If the chunk isn't in the top-k, the tool
   shows you nothing useful instead of inventing an answer.
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

- `.env` is gitignored and **never** committed. The key is read from the
  environment; `.env.example` ships as a placeholder.
- `data/`, `transcripts/`, `*.pdf` at the repo root, and `.cache/` are ignored.
- The repo ships no documents at all, by design.

## License

No license is granted. All rights reserved. The code may not be reused without
permission.

## Roadmap

- Answer synthesis with strict source citation (and an "I don't know" fallback)
- FAISS or sqlite-vec backed persistent vector index
- BM25 + vector hybrid retrieval; cross-encoder reranking
- GitHub Actions CI (pytest + lint)
- OCR-free page anchors; better table handling
