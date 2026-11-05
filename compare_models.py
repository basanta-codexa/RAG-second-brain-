from sentence_transformers import SentenceTransformer
import numpy as np

MODELS = [
    "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
]

question = "How do I make my system survive server crashes?"
candidates = [
    "Fault tolerance keeps a service running when machines fail.",      # should rank 1
    "Replication copies data to several nodes so one failure is safe.",  # should rank 2
    "Gradient descent minimizes the cost function.",
    # related topic, wrong answer
    "Load balancers spread traffic across servers.",
    "My cat likes to sleep on the sofa.",
]


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


for name in MODELS:
    model = SentenceTransformer(name)
    q = model.encode(question)
    scores = [(cosine(q, model.encode(c)), c) for c in candidates]
    scores.sort(reverse=True)
    print(f"\n=== {name} ===")
    for s, c in scores:
        print(f"{s:.3f}  {c}")
