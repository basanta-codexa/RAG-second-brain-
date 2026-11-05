import numpy as np
from sentence_transformers import SentenceTransformer
from loader import load_folder
from chunker import chunk_records

# ---- experiment settings: change these and rerun ----
MODEL_NAME = "BAAI/bge-base-en-v1.5"      # also try "BAAI/bge-base-en-v1.5"
# MiniLM reads ~1000 chars max, so keep this <= 800
CHUNK_SIZE = 800
OVERLAP = 150
USE_BGE_PREFIX = True                # set True when testing a bge model
TOP_K = 3
# -----------------------------------------------------

BGE_PREFIX = "Represent this sentence for searching relevant passages: "

print("Loading files and chunking...")
chunks = chunk_records(load_folder("data"), CHUNK_SIZE, OVERLAP)
print(f"{len(chunks)} chunks")

print(f"Loading model {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

print("Embedding all chunks (one time)...")
chunk_vectors = model.encode(
    [c["text"] for c in chunks],
    normalize_embeddings=True,
    show_progress_bar=True,
)


def search(question: str, top_k: int = TOP_K) -> list[dict]:
    query = (BGE_PREFIX + question) if USE_BGE_PREFIX else question
    q_vec = model.encode(query, normalize_embeddings=True)
    scores = chunk_vectors @ q_vec            # cosine similarity with every chunk
    best = np.argsort(scores)[::-1][:top_k]   # indices of the highest scores
    return [{**chunks[i], "score": float(scores[i])} for i in best]


if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit", ""):
            break
        for rank, r in enumerate(search(q), start=1):
            print(
                f"\n#{rank}  score {r['score']:.3f}  [{r['source']} | page {r['page']}]")
            print(r["text"][:300] + "...")
