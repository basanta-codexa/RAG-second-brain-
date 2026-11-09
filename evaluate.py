"""Evaluate retrieval quality on the labelled testset.

Honesty rules baked into this script:
- Tests whose source file is missing from --folder are SKIPPED and reported,
  never counted as passes or failures silently.
- Tests whose ground truth accepts "any page of the source" (page=None), or
  whose source only has one page, are counted as *lenient* and reported
  separately from page-specific tests. Headline metrics are also shown for the
  strict subset only.
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
        if not t.get("q") or not t.get("answers"):
            raise ValueError(f"malformed test item: {t!r}")
        for src, page in t["answers"]:
            if not isinstance(src, str) or not src:
                raise ValueError(f"bad source in {t['q']!r}")
            if page is not None and not isinstance(page, int):
                raise ValueError(f"bad page in {t['q']!r}")


def find_rank(results, answers) -> int | None:
    """Rank (1-based) of the first result matching any answer, else None.

    A test may name either an origin ("data/notes.md") or a bare filename.
    """
    for rank, r in enumerate(results, start=1):
        for src, page in answers:
            source_match = r["source"] == src or r.get("origin") == src
            if source_match and (page is None or r["page"] == page):
                return rank
    return None


def compute_metrics(ranks: list[int | None], top_k: int) -> dict:
    """Pure metric calculation; ranks are 1-based and already within top_k."""
    n = len(ranks)
    hit1 = sum(1 for r in ranks if r == 1)
    hitk = sum(1 for r in ranks if r is not None)
    reciprocal = sum(1.0 / r for r in ranks if r is not None)
    return {
        "n": n,
        "hit1": hit1,
        "hitk": hitk,
        "mrr": reciprocal / n if n else 0.0,
        "hit1_rate": hit1 / n if n else 0.0,
        "hitk_rate": hitk / n if n else 0.0,
    }


def _single_page_sources(index: dict) -> set[str]:
    max_page: dict[str, int] = {}
    for chunk in index["chunks"]:
        for key in (chunk.get("source"), chunk.get("origin")):
            if key:
                max_page[key] = max(max_page.get(key, 0), chunk["page"])
    return {key for key, page in max_page.items() if page <= 1}


def evaluate(folder: str, top_k: int, model_name: str,
             chunk_size: int, overlap: int, rebuild: bool,
             hybrid: bool = False, rerank: bool = False,
             fail_under: float = 0.0) -> int:
    validate_tests(TESTS)
    index = srch.create_index(folder, model_name, chunk_size, overlap, rebuild)
    if index["errors"]:
        for err in index["errors"]:
            print(f"INDEX ERROR: {err['file']}: {err['error']}")
    encoder = index["encoder"] or srch.load_encoder(model_name)

    evaluated, skipped = [], []
    for t in TESTS:
        sources = {src for src, _ in t["answers"]}
        missing = [src for src in sorted(sources)
                   if not (Path(folder) / Path(src).name).is_file()]
        if missing:
            skipped.append((t, missing))
        else:
            evaluated.append(t)

    print(f"tests: {len(TESTS)} total | evaluated: {len(evaluated)} | skipped (source missing): {len(skipped)}")
    mode = "hybrid (BM25 + vector, RRF)" if hybrid else "vector-only"
    if rerank:
        mode += " + cross-encoder rerank"
    print(f"mode: {mode}\n")

    single_page = _single_page_sources(index)
    lenient_tests = []
    strict_tests = []
    ranks: list[int | None] = []
    strict_ranks: list[int | None] = []
    misses: list[tuple[dict, list]] = []

    for t in evaluated:
        question_is_lenient = (
            any(page is None for _, page in t["answers"])
            or any(src in single_page for src, _ in t["answers"])
        )
        results = srch.query_index(index, encoder, t["q"], top_k,
                                   hybrid=hybrid, rerank=rerank)
        rank = find_rank(results, t["answers"])
        ranks.append(rank)
        if question_is_lenient:
            lenient_tests.append(t)
        else:
            strict_tests.append(t)
            strict_ranks.append(rank)
        if rank is None:
            misses.append((t, results))

    overall = compute_metrics(ranks, top_k)
    strict = compute_metrics(strict_ranks, top_k)
    print(f"overall: hit@1: {overall['hit1']}/{overall['n']}   "
          f"hit@{top_k}: {overall['hitk']}/{overall['n']}   MRR: {overall['mrr']:.3f}")
    if strict["n"]:
        print(f"strict:  hit@1: {strict['hit1']}/{strict['n']}   "
              f"hit@{top_k}: {strict['hitk']}/{strict['n']}   MRR: {strict['mrr']:.3f}")
    print(f"lenient tests (any page, or a single-page source): {len(lenient_tests)}")
    print(f"strict tests (page-specific, multi-page sources): {len(strict_tests)}\n")

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

    if fail_under and overall["hitk_rate"] < fail_under:
        print(f"\nFAIL: hit@{top_k} {overall['hitk_rate']:.3f} is below --fail-under {fail_under:.3f}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval hit rate and MRR.")
    parser.add_argument("--folder", default="data")
    parser.add_argument("--k", type=int, default=srch.DEFAULT_TOP_K)
    parser.add_argument("--model", default=srch.DEFAULT_MODEL)
    parser.add_argument("--chunk-size", type=int, default=srch.DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=srch.DEFAULT_OVERLAP)
    parser.add_argument("--rebuild", action="store_true",
                        help="drop the collection and re-embed everything")
    parser.add_argument("--hybrid", action="store_true",
                        help="enable BM25+vector fusion (measured regression on this corpus — see README)")
    parser.add_argument("--rerank", action="store_true",
                        help="re-rank top candidates with a cross-encoder")
    parser.add_argument("--fail-under", type=float, default=0.0,
                        help="exit 1 when hit@k is below this rate (0 disables)")
    args = parser.parse_args()
    try:
        return evaluate(args.folder, args.k, args.model, args.chunk_size, args.overlap,
                        args.rebuild, hybrid=args.hybrid, rerank=args.rerank,
                        fail_under=args.fail_under)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
