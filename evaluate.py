import search as s
from testset import TESTS


def is_hit(result, answers):
    for src, page in answers:
        if result["source"] == src and (page is None or result["page"] == page):
            return True
    return False


def run(top_k=3):
    hit1 = hit_k = 0
    misses = []
    for t in TESTS:
        results = s.search(t["q"], top_k)
        found = [i for i, r in enumerate(
            results, 1) if is_hit(r, t["answers"])]
        if found:
            hit_k += 1
            hit1 += found[0] == 1
        else:
            misses.append((t, results))

    print(f"Rank 1: {hit1}/{len(TESTS)}   Top {top_k}: {hit_k}/{len(TESTS)}\n")

    for t, results in misses:
        print("=" * 70)
        print("MISS:", t["q"])
        print("Expected:", t["answers"])
        for rank, r in enumerate(results, 1):
            print(
                f"  #{rank} {r['score']:.3f} [{r['source']} | p{r['page']}] {r['text'][:150]}...")


if __name__ == "__main__":
    run()
