from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, downloads once

sentences = [
    "How do I make my system survive server crashes?",
    "Fault tolerance keeps a service running when machines fail.",
    "Gradient descent minimizes the cost function.",
    "My cat likes to sleep on the sofa.",
]

vectors = model.encode(sentences)
print("Shape:", vectors.shape)          # (4, 384)
print("First 5 numbers of sentence 0:", vectors[0][:5])


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


print("\nSimilarity to the question:")
for s, v in zip(sentences[1:], vectors[1:]):
    print(f"{cosine(vectors[0], v):.3f}  {s}")
