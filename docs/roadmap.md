# Roadmap scope

This repo is a local, evidence-first document workbench. Keep it small: prefer
correctness, retrieval quality, and finishing the existing UI over adding new
product surfaces that need their own infrastructure.

## Active work

Ordered by the current plan. Effort is relative, not a commitment.

| Item | Priority | Effort |
| --- | --- | --- |
| Professional UI redesign | Very high | Medium |
| Document library and management | Very high | Medium |
| PDF/document preview | High | Medium |
| Search and evidence workspace | Very high | Medium |
| Conversation history | High | Medium |
| Better citation handling | Very high | Medium |
| Improved error/loading states | High | Low |
| Token-aware chunking experiments | High | Medium |
| Better retrieval evaluation | Very high | Medium |

The small, low-effort items are done first so the higher-priority redesign work
starts from a stable UI.

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
