# Chunking A/B — 2026-11-22

Commands (vector mode, `k=3`, real `BAAI/bge-base-en-v1.5` embeddings, local Qdrant):

```powershell
python evaluate.py --k 3 --chunk-mode char --json char.json
python evaluate.py --k 3 --chunk-mode token --token-size 256 --token-overlap 48 --json token.json
python evaluate.py --k 3 --chunk-mode char --json char_restored.json
```

The first `char` run used a different uploaded PDF than the last two runs, so it
is excluded from the comparison. Only `token` and `char_restored` saw the same
corpus (four files in `data/`; the uploaded PDF produced no readable text and
was skipped).

| Mode | Indexed points | Strict hit@1 | Strict hit@3 (95% CI) | Strict MRR | Overall hit@1 |
| --- | --- | --- | --- | --- | --- |
| `char` (restored) | 297 | **19/22** | **22/22** (0.851–1.000) | **0.932** | **37/45** |
| `token` (256/48) | 235 | 17/22 | 21/22 (0.782–0.992) | 0.864 | 34/45 |

Token chunks are larger (~256 tokens vs ~800 characters), so the index has
fewer points and retrieval precision drops. The confidence intervals overlap;
with 22 strict cases a two-case difference is not strong evidence either way.
The default stays `char`.

Note: tokenizing a whole page to obtain offsets emits a harmless
"sequence length longer than 512" warning from the tokenizer; offsets are still
returned for the full text.

The live index was restored to `char` after the experiment.
