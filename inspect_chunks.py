import sys
from loader import load_folder
from chunker import chunk_records

chunks = chunk_records(load_folder("data"), 800, 150)
if len(sys.argv) < 3:
    print('Usage: python inspect_chunks.py find "term"  |  page FILE PAGE')
    sys.exit(1)
mode = sys.argv[1]

if mode == "find":          # python inspect_chunks.py find "cap theorem"
    term = sys.argv[2].lower()
    found = [c for c in chunks if term in c["text"].lower()]
    print(f"{len(found)} chunks contain '{term}'\n")
    for c in found[:5]:
        print(f"[{c['source']} | p{c['page']}]")
        print(c["text"][:400], "\n" + "-" * 40)

elif mode == "page":        # python inspect_chunks.py page system-design-notes.pdf 12
    src, page = sys.argv[2], int(sys.argv[3])
    for c in chunks:
        if c["source"] == src and c["page"] == page:
            print(c["text"], "\n" + "-" * 40)
