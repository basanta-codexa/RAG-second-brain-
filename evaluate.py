"""Evaluate retrieval quality on the labelled testset.

Honesty rules baked into this script:
- Tests whose source file is missing from --folder are SKIPPED and reported,
  never counted as passes or failures silently.
- Tests whose ground truth accepts "any page of the source" (page=None) are
  counted as *lenient* and reported separately from page-specific tests.
- Metrics reported: hit@1, hit@k, MRR (over evaluated tests).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import search as srch
from testset import TESTS


def validate_tests(tests: list[dict]) -> None:
    for t in tests:
        assert t.get("q") and t.get("answers"), f"malformed test item: {t!r}"
        for src, page in t["answers"]:
            assert isinstance(src, str) and src, f"bad source in {t['q']!r}"
            assert page is None or isinstance(page, int), f"bad page in {t['q']!r}"


def find_rank(results, answers) -> int | None:
    """Rank (1-based) of the first result matching any answer, else None."""
    for rank, r in enumerate(results, start=1):
        for src, page in answers:
            if r["source"] == src and (page is None or r["page"] == page):
                return rank
    return None


def evaluate(folder: str, top_k: int, model_name: str,
             chunk_size: int, overlap: int, rebuild: bool,
             hybrid: bool = False, rerank: bool = False) -> int:
    validate_tests(TESTS)
    index = srch.create_index(folder, model_name, chunk_size, overlap, rebuild)
    encoder = index["encoder"] or srch.SentenceTransformer(model_name)

    evaluated, skipped = [], []
    for t in TESTS:
        sources = {src for src, _ in t["answers"]}
        missing = [src for src in sorted(sources)
                   if not (Path(folder) / src).is_file()]
        if missing:
            skipped.append((t, missing))
        else:
            evaluated.append(t)

    print(f"tests: {len(TESTS)} total | evaluated: {len(evaluated)} | skipped (source missing): {len(skipped)}")
    mode = "hybrid (BM25 + vector, RRF)" if hybrid else "vector-only"
    if rerank:
        mode += " + cross-encoder rerank"
    print(f"mode: {mode}\n")

    reciprocal, hit1, hitk = 0.0, 0, 0
    misses: list[tuple[dict, list]] = []
    lenient = 0
    for t in evaluated:
        question_is_lenient = any(page is None for _, page in t["answers"])
        lenient += question_is_lenient
        results = srch.query_index(index, encoder, t["q"], top_k,
                                   hybrid=hybrid, rerank=rerank)
        rank = find_rank(results, t["answers"])
        if rank is not None:
            hitk += 1
            hit1 += rank == 1
            reciprocal += 1.0 / rank
        else:
            misses.append((t, results))

    n = len(evaluated)
    print(f"hit@1: {hit1}/{n}   hit@{top_k}: {hitk}/{n}   MRR: {reciprocal / n if n else 0:.3f}")
    print(f"lenient tests inside evaluated set (any page accepted): {lenient}\n")

    for t, missing in skipped:
        print(f"SKIP: {t['q']}")
        print(f"      missing source(s): {', '.join(missing)}")

    if skipped:
        print()

    for t, results in misses:
        print("=" * 70)
        print("MISS:", t["q"])
        print("Expected:", t["answers"])
        for rank, r in enumerate(results, 1):
            print(f"  #{rank} {r['score']:.3f} [{r['source']} | p{r['page']}] {r['text'][:150]}...")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval hit rate and MRR.")
    parser.add_argument("--folder", default="data")
    parser.add_argument("--k", type=int, default=srch.DEFAULT_TOP_K)
    parser.add_argument("--model", default=srch.DEFAULT_MODEL)
    parser.add_argument("--chunk-size", type=int, default=srch.DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=srch.DEFAULT_OVERLAP)
    parser.add_argument("--rebuild", action="store_true", help="ignore the embedding cache and re-embed")
    parser.add_argument("--hybrid", action="store_true", help="enable BM25+vector fusion (measured regression on this corpus — see README)")
    parser.add_argument("--rerank", action="store_true", help="re-rank top candidates with a cross-encoder")
    args = parser.parse_args()
    return evaluate(args.folder, args.k, args.model, args.chunk_size, args.overlap, args.rebuild,
                    hybrid=args.hybrid, rerank=args.rerank)


if __name__ == "__main__":
    sys.exit(main())
