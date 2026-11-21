# Roadmap scope

This repo is a local, evidence-first document workbench. Keep it small: prefer
correctness, retrieval quality, and finishing the existing UI over adding new
product surfaces that need their own infrastructure.

## Active work

Items from `improvement.md` that are in scope:

- **UI/UX**: dashboard, search history, chat with conversation memory,
  citation side panel, settings page, better indexing progress, source
  inspection, multi-file upload.
- **Document management**: detail/preview, duplicate detection, per-file
  processing status, manual re-index, manual tags/metadata.
- **Retrieval**: token-aware chunking, metadata filtering, context-aware
  retrieval, context compression, better citation validation, reranking
  evaluation.
- **Ingestion**: HTML, CSV, generic image OCR through the existing
  transcription path.

## Deferred — non-goals for now

These are explicitly out of scope. Do not add code, dependencies, or data
schemas for them:

| Item | Why deferred |
| --- | --- |
| Knowledge graph | Interesting, but not necessary |
| Multi-query retrieval | Experiment before adopting |
| Query rewriting | Useful for difficult queries |
| Automatic tagging | Useful, but not essential |
| YouTube ingestion | Good future feature |
| DOCX support | Reasonable addition |
| Web clipping | Useful later |
| Cross-document reasoning | Valuable after improving citations |

Revisit only after the active items are done and a concrete need exists.
