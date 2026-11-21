# Retrieval evaluation

Scope: 45 of 45 test cases, measured 2026-11-17. At that time the system-design
source in `data/` was built from 26 cached transcript pages rather than the full
39-page PDF; it has since been expanded. These numbers are therefore historical
and not directly comparable to the current corpus. All 12 cases that referenced
it pointed to pages 2–15, so their source material was present. Re-run
`evaluate.py` after any corpus change.

## Setup

- `k = 3`; chunk size 800, overlap 150.
- Embeddings: `BAAI/bge-base-en-v1.5`. Default reranker:
  `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Real models from the local Hugging Face cache. Qdrant at `127.0.0.1:6333`,
  collection `second_brain`, 283 points from 4 sources.
- 22 strict cases (page-specific, multi-page source); 23 lenient cases (any
  page, or single-page source).

## Results

| Mode | Overall hit@1 | Overall hit@3 | Overall MRR | Strict hit@1 | Strict hit@3 | Strict MRR |
| --- | --- | --- | --- | --- | --- | --- |
| `vector` | 38/45 | **45/45** | 0.919 | **19/22** | **22/22** | **0.932** |
| `hybrid` | 38/45 | 44/45 | 0.904 | 16/22 | 21/22 | 0.826 |
| `rerank` | 36/45 | 44/45 | 0.874 | 15/22 | 21/22 | 0.795 |
| `hybrid+rerank` | 36/45 | 44/45 | 0.874 | 15/22 | 21/22 | 0.795 |

Additional fusion strategies using the default reranker:

| Strategy | Overall hit@1 | Overall MRR | Strict hit@1 | Strict MRR |
| --- | --- | --- | --- | --- |
| Blend, cross-encoder weight 0.15 | 38/45 | 0.911 | 19/22 | 0.932 |
| Rank fusion (RRF) | **39/45** | **0.922** | 18/22 | 0.909 |
| Cross-encoder, pool 5 | 36/45 | 0.878 | 15/22 | 0.818 |
| Cross-encoder, pool 10 | 36/45 | 0.874 | 15/22 | 0.795 |

RRF gains one overall case over vector and loses one strict case. It still pays
the cross-encoder's latency, so it is not the default.

## Reranker behavior

| Diagnostic, strict cases | Result |
| --- | --- |
| Gold page in vector top 50 | 22/22 |
| Gold page in hybrid top 50 | 22/22 |
| Vector #1 demoted by the cross-encoder | 5/22 |
| Strict hit@1, vector vs pure rerank | 19/22 vs 15/22 |
| Strict hit@3, vector vs pure rerank | 22/22 vs 21/22 |

Candidate recall is not the problem. Pools of 5, 10, 20, and 50 produce the same
strict hit@1 of 15/22. The cross-encoder reorders chunks from the same correct
page; `hit@1` credits only the first one. On the prose-heavy system-design pages
it mostly agrees with vector, so the loss is concentrated in the lecture notes.
That fits a model trained for short web passages scoring 800-character technical
chunks.

## Latency

Measured before `system-design-notes.md` was added, on a 253-chunk index with
warm models and warm cache, 20 queries per mode. Query encoding is included in
every mode; cross-encoder inference only where reranking is used. Relative
ordering is unchanged at 283 chunks.

| Mode | ms/query |
| --- | --- |
| `vector` | **46.9** |
| `hybrid` | 51.8 |
| `rerank` | 1165.7 |
| `hybrid+rerank` | 1099.7 |

The cached BM25 view adds about 5 ms over vector search. The cross-encoder adds
roughly 1.05 s and dominates query time.

## Alternate reranker

`BAAI/bge-reranker-base` is selectable through `SECOND_BRAIN_RERANKER` and
`--reranker`. Measured on the same cases and corpus:

| Strategy | Overall hit@1 | Overall MRR | Strict hit@1 | Strict MRR |
| --- | --- | --- | --- | --- |
| `vector`, no reranker | 38/45 | 0.919 | 19/22 | 0.932 |
| Pure rerank | 38/45 | 0.915 | 16/22 | 0.848 |
| Hybrid + rerank | 39/45 | 0.930 | 17/22 | 0.879 |
| Blend, weight 0.15 | 39/45 | 0.926 | 20/22 | 0.947 |
| Rank fusion | 39/45 | 0.930 | **20/22** | **0.955** |

Pure reranking still loses three strict cases. Blending and rank fusion gain one
strict case and improve MRR, but a single case in 22 is not a reliable
difference. Cross-encoder time for this model is about 8.4 s per 20-candidate
pool, versus 1.05 s for the default reranker and 47 ms for vector search. The
option remains available for experimentation, not as a recommendation.

## Conclusion

`hybrid+rerank` ties for the lowest accuracy among the four modes and costs about
23× more per query than vector. The alternate reranker can gain one strict case
when blended, at roughly 8.4 s per query. The default stays `vector`.

## Reproduce

```powershell
python evaluate.py --k 3
python evaluate.py --k 3 --hybrid
python evaluate.py --k 3 --rerank
python evaluate.py --k 3 --hybrid-rerank
python evaluate.py --k 3 --rerank --reranker BAAI/bge-reranker-base
```

The corpus, Qdrant snapshot, and model cache are not shipped with the repo, so
these commands reproduce the method, not the author's exact numbers.
