from loader import load_folder


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks, preferring to cut at sentence ends."""
    text = " ".join(text.split())  # collapse messy whitespace/newlines
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        # try to end at a sentence boundary instead of mid-sentence
        if end < len(text):
            cut = text.rfind(". ", start, end)
            if cut > start + chunk_size // 2:  # only if it's not too early
                end = cut + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = end - overlap  # step back to create the overlap
    return chunks


def chunk_records(records: list[dict], chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    """Turn page-level records into chunk-level records, keeping metadata."""
    all_chunks = []
    for rec in records:
        for i, piece in enumerate(chunk_text(rec["text"], chunk_size, overlap)):
            all_chunks.append({
                "text": piece,
                "source": rec["source"],
                "page": rec["page"],
                "chunk_id": f"{rec['source']}-p{rec['page']}-c{i}",
            })
    return all_chunks


if __name__ == "__main__":
    CHUNK_SIZE = 300   # <- change only this line to experiment
    OVERLAP = 150

    records = load_folder("data")
    chunks = chunk_records(records, CHUNK_SIZE, OVERLAP)
    print(f"{len(records)} pages -> {len(chunks)} chunks")
    lengths = [len(c["text"]) for c in chunks]
    print(
        f"Chunk length: min {min(lengths)}, max {max(lengths)}, avg {sum(lengths)//len(lengths)}")
    print()
    for c in chunks[:3]:
        print(f"[{c['chunk_id']}]")
        print(c["text"])
        print("-" * 40)
